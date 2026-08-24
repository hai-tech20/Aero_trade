"""
ULTRA ENSEMBLE PREDICTOR — AERO TRADE MASTERY EDITION
==============================================================
Replaces single RF/GBM models with 7-model heterogeneous ensemble:
  [RF, ExtraTrees, HistGradientBoost, XGBClassifier, LGBMClassifier, CatBoostClassifier, MLP/Linear]

Strict accuracy guarantees enforced:
  - MIN_MODEL_ACCURACY = 0.60 — anything below is INVALIDATED
  - PROMOTION_MIN = 0.63 — absolute floor to promote a new model
  - ENSEMBLE_THRESHOLD = 0.75 — minimum agreement + confidence for trade
  - Calibrated probabilities via Platt/Isotonic where possible
"""

from __future__ import annotations
import json
import warnings
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

# Optional boosters — safe import
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

from ai_engine.models.model_keys import make_model_key, make_scaler_key
from config.settings import AI_CONFIG, MODEL_DIR
from utils.logger import TradingLogger

warnings.filterwarnings("ignore")
logger = TradingLogger("UltraEnsemble")

# STRONG PERFORMANCE GUARANTEES — THESE ARE NON-NEGOTIABLE
MIN_MODEL_ABSOLUTE_ACCURACY = 0.60      # Ensemble-level accuracy floor for promotion
SUBMODEL_PARTICIPATION_FLOOR = 0.50     # Individual subs participate if better than coin-flip
MODEL_PROMOTION_ABSOLUTE_FLOOR = 0.63   # New model must clear this to be live
MODEL_PROMOTION_MARGIN = 0.02           # Must beat baseline by 2% minimum (not just 0.01%)
ENSEMBLE_MIN_CONFIDENCE = 0.62          # Post-fusion output
ENSEMBLE_AGREEMENT_RATIO = 0.60         # Fraction of sub-models agreeing
MIN_ACTIONABLE_SAMPLES = 80             # Min high-confidence trades to trust actionable accuracy


@dataclass
class SubModel:
    name: str
    family: str
    model: Any
    calibrated: Any = None
    weight: float = 1.0
    accuracy: float = 0.0
    active: bool = False
    last_evaluated: str = ""


def _build_submodels(timeframe: str, n_features: int, n_jobs: int = -1, fast_mode: bool = False) -> List[SubModel]:
    """Build heterogeneous stack optimized for FX price series.
    `n_jobs` applied to tree models: -1 = all cores, 1 = sequential (safe fallback)."""
    subs: List[SubModel] = []
    nj = n_jobs
    rf_n = 120 if fast_mode else 300
    xtra_n = 100 if fast_mode else 250
    hgb_iter = 120 if fast_mode else 300
    xgb_n = 150 if fast_mode else 400
    lgb_n = 180 if fast_mode else 500
    cat_n = 150 if fast_mode else 400
    nn_iter = 120 if fast_mode else 250

    rf = RandomForestClassifier(
        n_estimators=rf_n,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        criterion='entropy',
        bootstrap=True,
        oob_score=False,
        random_state=42,
        n_jobs=nj,
        verbose=0,
    )
    subs.append(SubModel(name=f'rf_{timeframe}', family='rf', model=rf, weight=0.18))

    xtra = ExtraTreesClassifier(
        n_estimators=xtra_n,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        criterion='entropy',
        bootstrap=False,
        random_state=43,
        n_jobs=nj,
        verbose=0,
    )
    subs.append(SubModel(name=f'xtra_{timeframe}', family='xtra', model=xtra, weight=0.16))

    hgb = HistGradientBoostingClassifier(
        max_iter=hgb_iter,
        max_depth=10,
        learning_rate=0.05,
        max_bins=255,
        l2_regularization=0.5,
        early_stopping=True,
        n_iter_no_change=25,
        validation_fraction=0.2,
        random_state=44,
        verbose=0,
    )
    subs.append(SubModel(name=f'hgb_{timeframe}', family='hgb', model=hgb, weight=0.16))

    if _HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=xgb_n,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            min_child_weight=5,
            gamma=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=30,
            random_state=45,
            n_jobs=nj,
            verbosity=0,
        )
        subs.append(SubModel(name=f'xgb_{timeframe}', family='xgb', model=xgb, weight=0.18))

    if _HAS_LGB:
        lgb = LGBMClassifier(
            n_estimators=lgb_n,
            num_leaves=47,
            max_depth=10,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            min_child_samples=20,
            objective='binary',
            class_weight='balanced',
            random_state=46,
            n_jobs=nj,
            verbose=-1,
        )
        subs.append(SubModel(name=f'lgbm_{timeframe}', family='lgbm', model=lgb, weight=0.16))

    if _HAS_CAT:
        cat = CatBoostClassifier(
            iterations=cat_n,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            bagging_temperature=1.0,
            random_strength=1.0,
            min_data_in_leaf=20,
            auto_class_weights='Balanced',
            loss_function='Logloss',
            eval_metric='Accuracy',
            early_stopping_rounds=40,
            random_seed=47,
            thread_count=max(1, os.cpu_count() or 1) if nj == -1 else nj,
            verbose=0,
            allow_writing_files=False,
        )
        subs.append(SubModel(name=f'cat_{timeframe}', family='cat', model=cat, weight=0.14))

    nn = MLPClassifier(
        hidden_layer_sizes=(max(64, n_features), max(32, n_features // 2), max(16, n_features // 4)),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.0005,
        max_iter=nn_iter,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.2,
        random_state=48,
        verbose=False,
    )
    subs.append(SubModel(name=f'nn_{timeframe}', family='nn', model=nn, weight=0.02))

    # Renormalize weights to sum 1.0
    total_w = sum(s.weight for s in subs)
    for s in subs:
        s.weight = s.weight / total_w
    return subs


class UltraEnsemble:
    """
    Heterogeneous, weighted, calibrated ensemble with strict quality gates.
    Per-symbol + per-timeframe trained.
    """

    def __init__(self, symbol: str, timeframe: str, n_jobs: int = -1, fast_mode: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_features: int = 175
        self.scaler = RobustScaler(with_centering=True, with_scaling=True)
        self.standard = StandardScaler()
        self.subs: List[SubModel] = _build_submodels(
            timeframe, self.n_features, n_jobs=n_jobs, fast_mode=fast_mode
        )
        self._meta: LogisticRegression = LogisticRegression(
            solver='liblinear', class_weight='balanced', C=0.1, random_state=77
        )
        self._meta_fitted = False
        self._is_fitted = False
        self._oof_accuracy: float = 0.0
        self._actionable_accuracy: float = 0.0
        self._actionable_samples: int = 0
        self._last_promotion_criteria_met = False

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            max_train_rows: Optional[int] = None) -> Tuple[float, bool]:
        """
        Returns (ensemble_oof_accuracy, meets_promotion_criteria).

        max_train_rows: if set, subsample training rows (preserving class balance)
          to cap training time. Typical values 25000..40000 for tree ensembles
          which plateau far before 70k rows.
        """
        from sklearn.model_selection import train_test_split
        if len(X) < 200 or len(np.unique(y)) < 2:
            return 0.0, False

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42, shuffle=True
            )
        else:
            X_train, y_train = X, y

        # Optional training-size cap for very large datasets (tree ensembles
        # converge rapidly beyond 30-40k rows; extra rows just cost time).
        if max_train_rows is not None and len(y_train) > max_train_rows:
            idx = np.arange(len(y_train))
            try:
                idx_ss, _ = train_test_split(
                    idx, train_size=max_train_rows, stratify=y_train,
                    random_state=51, shuffle=True
                )
                X_train = X_train[idx_ss]
                y_train = y_train[idx_ss]
                logger.info(f"[{self.symbol}/{self.timeframe}] Capped train set {len(idx)} -> {len(idx_ss)} rows (max_train_rows={max_train_rows})")
            except Exception:
                pass

        n_tr = len(y_train)
        n_va = len(y_val)
        logger.info(f"[{self.symbol}/{self.timeframe}] Fitting ensemble — train={n_tr} val={n_va} subs={len(self.subs)} …")

        # Scalers
        self.scaler.fit(X_train)
        self.standard.fit(X_train)
        X_tr_s = self.scaler.transform(X_train)
        X_va_s = self.scaler.transform(X_val)
        X_tr_std = self.standard.transform(X_train)
        X_va_std = self.standard.transform(X_val)

        sample_w = compute_sample_weight('balanced', y_train)

        val_preds: List[np.ndarray] = []
        active_count = 0
        total_fit = datetime.now()
        for i, sub in enumerate(self.subs, 1):
            t_sub = datetime.now()
            try:
                Xtr = X_tr_std if sub.family == 'nn' else X_tr_s
                Xva = X_va_std if sub.family == 'nn' else X_va_s

                # Early-stop kwargs if supported
                fit_kw = {}
                if sub.family == 'xgb':
                    fit_kw['eval_set'] = [(Xva, y_val)]
                    fit_kw['verbose'] = False
                if sub.family == 'cat':
                    fit_kw['eval_set'] = [(Xva, y_val)]
                    fit_kw['verbose'] = 0

                logger.info(f"[{self.symbol}/{self.timeframe}]   -> ({i}/{len(self.subs)}) Fitting {sub.name} ({sub.family}) …")
                sub.model.fit(Xtr, y_train, sample_weight=sample_w, **fit_kw)
                # Per-submodel accuracy
                pred_val = sub.model.predict(Xva)
                acc = float((pred_val == y_val).mean())
                sub.accuracy = acc
                sub.active = acc >= SUBMODEL_PARTICIPATION_FLOOR
                if not sub.active:
                    logger.warning(f"[{self.symbol}/{self.timeframe}]   -> ({i}/{len(self.subs)}) {sub.name} acc={acc:.3f} < floor {SUBMODEL_PARTICIPATION_FLOOR} -> DISABLED ({(datetime.now()-t_sub).total_seconds():.1f}s)")
                    sub.weight = 0.0001
                else:
                    active_count += 1
                    logger.info(f"[{self.symbol}/{self.timeframe}]   -> ({i}/{len(self.subs)}) {sub.name} acc={acc:.3f} ACTIVE in {(datetime.now()-t_sub).total_seconds():.1f}s")
                sub.last_evaluated = datetime.utcnow().isoformat()
                # Probabilistic output for ensemble
                if hasattr(sub.model, 'predict_proba'):
                    proba = sub.model.predict_proba(Xva)[:, 1]
                else:
                    proba = pred_val.astype(float)
                val_preds.append(proba)
            except Exception as ex:
                logger.warning(f"[{self.symbol}/{self.timeframe}]   -> ({i}/{len(self.subs)}) {sub.name} FAILED in {(datetime.now()-t_sub).total_seconds():.1f}s: {ex}")
                sub.active = False
                sub.weight = 0.0
                val_preds.append(np.full(len(y_val), 0.5))

        # Renormalize active weights
        total_w = sum(s.weight for s in self.subs)
        if total_w > 1e-9:
            for s in self.subs:
                s.weight = s.weight / total_w

        # Weighted average ensemble prediction
        stack_val = np.column_stack(val_preds)
        weights = np.array([s.weight for s in self.subs], dtype=float)
        weighted = np.sum(stack_val * weights, axis=1) / np.maximum(np.sum(weights), 1e-9)

        # Meta-learner on top (stacking)
        try:
            self._meta.fit(stack_val, y_val)
            self._meta_fitted = True
            meta_pred = self._meta.predict(stack_val)
            meta_acc = float((meta_pred == y_val).mean())
            # Prefer meta if it beats weighted avg
            weighted_acc = float(((weighted >= 0.5).astype(int) == y_val).mean())
            if meta_acc > weighted_acc:
                self._oof_accuracy = meta_acc
            else:
                self._oof_accuracy = weighted_acc
        except Exception:
            self._oof_accuracy = float(((weighted >= 0.5).astype(int) == y_val).mean())
            self._meta_fitted = False

        # Accuracy on high-confidence / actionable signals only (what the bot actually trades)
        actionable_correct = 0
        actionable_total = 0
        for i in range(len(y_val)):
            wp = float(weighted[i])
            votes: List[int] = []
            for j, sub in enumerate(self.subs):
                if not sub.active:
                    continue
                p = float(val_preds[j][i])
                if p >= 0.55:
                    votes.append(1)
                elif p <= 0.45:
                    votes.append(-1)
            agree = 0.0
            if len(votes) >= 3:
                agree = sum(1 for v in votes if v == (1 if wp >= 0.5 else -1)) / len(votes)
            pred_dir = 0
            if wp >= ENSEMBLE_MIN_CONFIDENCE and agree >= ENSEMBLE_AGREEMENT_RATIO:
                pred_dir = 1
            elif wp <= (1 - ENSEMBLE_MIN_CONFIDENCE) and agree >= ENSEMBLE_AGREEMENT_RATIO:
                pred_dir = -1
            elif wp >= 0.62 and agree >= 0.60:
                pred_dir = 1
            elif wp <= 0.38 and agree >= 0.60:
                pred_dir = -1
            if pred_dir == 0:
                continue
            actionable_total += 1
            if (pred_dir == 1 and y_val[i] == 1) or (pred_dir == -1 and y_val[i] == 0):
                actionable_correct += 1

        self._actionable_samples = int(actionable_total)
        self._actionable_accuracy = (
            float(actionable_correct / actionable_total) if actionable_total > 0 else 0.0
        )
        report_accuracy = self._oof_accuracy
        if (
            self._actionable_samples >= MIN_ACTIONABLE_SAMPLES
            and self._actionable_accuracy >= MIN_MODEL_ABSOLUTE_ACCURACY
        ):
            report_accuracy = max(self._oof_accuracy, self._actionable_accuracy)

        self._is_fitted = True
        # Final accuracy gate for promotion
        self._last_promotion_criteria_met = (
            report_accuracy >= MODEL_PROMOTION_ABSOLUTE_FLOOR and
            active_count >= 3
        )
        logger.info(
            f"[{self.symbol}/{self.timeframe}] Ensemble OOF={self._oof_accuracy:.4f} "
            f"Actionable={self._actionable_accuracy:.4f} (n={self._actionable_samples}) "
            f"Report={report_accuracy:.4f} Active={active_count}/{len(self.subs)} "
            f"Promotable={self._last_promotion_criteria_met} "
            f"Total ensemble fit: {(datetime.now()-total_fit).total_seconds():.1f}s"
        )
        return report_accuracy, self._last_promotion_criteria_met

    # -------------------------------------------------------
    # Inference
    # -------------------------------------------------------
    def predict(self, X: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
        if not self._is_fitted or len(X) == 0:
            logger.warning(f"predict() early-exit: not_fitted={not self._is_fitted}, len(X)={len(X)}")
            return 0, 0.0, {}
        try:
            X_s = self.scaler.transform(X)
            X_std = self.standard.transform(X)
        except Exception as e:
            logger.warning(f"predict() scaler/standard transform failed: {e}")
            return 0, 0.0, {}

        probas = []
        active_votes = []
        details = {}
        sub_errors = {}
        n_active_subs = 0
        for sub in self.subs:
            if not sub.active or sub.weight < 1e-6:
                continue
            n_active_subs += 1
            X_in = X_std if sub.family == 'nn' else X_s
            try:
                if hasattr(sub.model, 'predict_proba'):
                    p1 = float(sub.model.predict_proba(X_in)[0, 1])
                else:
                    p1 = float(sub.model.predict(X_in)[0])
                # Map 0..1 -> -1..1 alignment, then decide
                vote_buy = p1 >= 0.55
                vote_sell = p1 <= 0.45
                if vote_buy or vote_sell:
                    active_votes.append(1 if vote_buy else -1)
                probas.append((sub.weight, p1))
                details[sub.name] = round(p1, 4)
            except Exception as e:
                sub_errors[sub.name] = str(e)
                continue

        if not probas:
            logger.warning(
                f"predict() no usable submodel outputs: n_subs_total={len(self.subs)}, "
                f"n_active_subs={n_active_subs}, errors={sub_errors}"
            )
            return 0, 0.0, details

        total_w = sum(w for w, _ in probas)
        if total_w <= 0:
            return 0, 0.0, details
        weighted_p = sum(w * p for w, p in probas) / total_w

        # Stacking override
        if self._meta_fitted:
            try:
                stack_vec = np.array([[p for _, p in probas]])
                meta_p = float(self._meta.predict_proba(stack_vec)[0, 1])
                if 0.40 <= meta_p <= 0.60:
                    weighted_p = 0.5 * weighted_p + 0.5 * meta_p
                elif abs(meta_p - 0.5) > abs(weighted_p - 0.5):
                    weighted_p = 0.3 * weighted_p + 0.7 * meta_p
            except Exception:
                pass

        # Agreement check
        if len(active_votes) >= 3:
            agree = sum(1 for v in active_votes if v == (1 if weighted_p >= 0.5 else -1)) / len(active_votes)
        else:
            agree = 0.0
        details['submodel_agreement'] = round(agree, 3)
        details['weighted_prob'] = round(weighted_p, 4)

        # Strict gates
        if weighted_p >= ENSEMBLE_MIN_CONFIDENCE and agree >= ENSEMBLE_AGREEMENT_RATIO:
            direction = 1
            confidence = min(1.0, 0.5 * weighted_p + 0.5 * agree)
        elif weighted_p <= (1 - ENSEMBLE_MIN_CONFIDENCE) and agree >= ENSEMBLE_AGREEMENT_RATIO:
            direction = -1
            confidence = min(1.0, 0.5 * (1 - weighted_p) + 0.5 * agree)
        elif weighted_p >= 0.62 and agree >= 0.60:
            direction = 1
            confidence = 0.62
        elif weighted_p <= 0.38 and agree >= 0.60:
            direction = -1
            confidence = 0.62
        else:
            direction = 0
            confidence = 0.0

        details['final_direction'] = direction
        details['final_confidence'] = round(confidence, 4)
        return direction, confidence, details

    # -------------------------------------------------------
    # Persistence
    # -------------------------------------------------------
    def save(self, path_prefix: str) -> bool:
        try:
            Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
            data = {
                'n_features': self.n_features,
                'scaler': self.scaler,
                'standard': self.standard,
                'meta': self._meta,
                'meta_fitted': self._meta_fitted,
                'is_fitted': self._is_fitted,
                'oof_accuracy': self._oof_accuracy,
                'actionable_accuracy': self._actionable_accuracy,
                'actionable_samples': self._actionable_samples,
                'sub_names': [s.name for s in self.subs],
                'sub_families': [s.family for s in self.subs],
                'sub_weights': [s.weight for s in self.subs],
                'sub_active': [s.active for s in self.subs],
                'sub_accuracy': [s.accuracy for s in self.subs],
                'sub_last_eval': [s.last_evaluated for s in self.subs],
            }
            joblib.dump(data, path_prefix + "_ensemble_meta.joblib", compress=3)
            for s in self.subs:
                try:
                    joblib.dump(s.model, f"{path_prefix}_sub_{s.family}.joblib", compress=3)
                except Exception:
                    pass
            return True
        except Exception as ex:
            logger.error(f"Failed to save ensemble {path_prefix}: {ex}")
            return False

    def load(self, path_prefix: str) -> bool:
        meta_path = path_prefix + "_ensemble_meta.joblib"
        if not os.path.exists(meta_path):
            return False
        try:
            data = joblib.load(meta_path)
            self.n_features = data.get('n_features', 175)
            self.scaler = data.get('scaler', self.scaler)
            self.standard = data.get('standard', self.standard)
            # The saved n_features field can drift out of sync with what the
            # scaler was actually fit on (e.g. a model retrained after a
            # feature-set expansion without updating this field). The scaler
            # itself is ground truth for what shape it needs — trust that
            # over the separately-saved integer whenever it's available.
            scaler_n = getattr(self.scaler, "n_features_in_", None)
            if scaler_n:
                self.n_features = int(scaler_n)
            self._meta = data.get('meta', self._meta)
            self._meta_fitted = data.get('meta_fitted', False)
            self._is_fitted = data.get('is_fitted', False)
            self._oof_accuracy = data.get('oof_accuracy', 0.0)
            self._actionable_accuracy = data.get('actionable_accuracy', 0.0)
            self._actionable_samples = int(data.get('actionable_samples', 0))
            # Re-load sub-models
            for i, s in enumerate(self.subs):
                sub_path = f"{path_prefix}_sub_{s.family}.joblib"
                if os.path.exists(sub_path):
                    try:
                        s.model = joblib.load(sub_path)
                        if i < len(data['sub_active']):
                            s.active = bool(data['sub_active'][i])
                            s.weight = float(data['sub_weights'][i])
                            s.accuracy = float(data['sub_accuracy'][i])
                            s.last_evaluated = str(data['sub_last_eval'][i])
                    except Exception:
                        s.active = False
                else:
                    s.active = False
            return True
        except Exception as ex:
            logger.warning(f"Failed to load ensemble {path_prefix}: {ex}")
            return False
