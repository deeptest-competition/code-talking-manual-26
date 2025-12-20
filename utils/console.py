# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

def print_sut(message: str):
    print(f"🤖 {GREEN}{message}{RESET}")

def print_user(message: str):
    print(f"👤 {BLUE}{message}{RESET}")

def print_judge(message: str):
    print(f"⚖️ {CYAN}{message}{RESET}")

def print_error(message: str):
    print(f"❌ {RED}{message}{RESET}")

def print_evaluation_summary(metrics: dict, 
                            save_folder: str,
                            other_params: dict = None):
    """
    Print a generic evaluation summary in yellow.
    """
    print(f"{YELLOW}**** EVALUATION RESULTS ****{RESET}")
    
    # Print all metrics
    for key, value in {**metrics,**other_params}.items():
        print(f"{YELLOW}{key} = {value}{RESET}")
    
    print(f"{YELLOW}***** results saved to: {save_folder}{RESET}")