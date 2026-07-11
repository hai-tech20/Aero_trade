class TradingLogger:
    def __init__(self, name):
        self.name = name

    def info(self, message):
        print(f"[INFO] [{self.name}] {message}")

    def warning(self, message):
        print(f"[WARNING] [{self.name}] {message}")

    def error(self, message):
        print(f"[ERROR] [{self.name}] {message}")

    def success(self, message):
        print(f"[SUCCESS] [{self.name}] {message}")
    
    def debug(self, message):
        print(f"[DEBUG] [{self.name}] {message}")
    
    def trade(self, message):
        print(f"[TRADE] [{self.name}] {message}")

# Global logger instance
logger = TradingLogger("System")