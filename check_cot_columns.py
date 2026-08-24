import cot_reports as cot

df = cot.cot_all(cot_report_type="legacy_fut")
print(df.columns.tolist())