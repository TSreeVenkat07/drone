import time
import os
import winsound

def play_beeps():
    print("Epoch 30 completed and evaluated! Playing notification beeps...")
    for _ in range(5):
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        time.sleep(0.2)
        winsound.Beep(1500, 300)  # Frequency: 1500Hz, Duration: 300ms
        time.sleep(0.2)

def monitor():
    eval_file = "venkateval.txt"
    print(f"Monitoring '{eval_file}' for Epoch 30 completion and evaluation...")
    
    while True:
        if os.path.exists(eval_file):
            try:
                with open(eval_file, "r") as f:
                    content = f.read()
                if "Epoch: 30 (completed)" in content or "Routine Eval (Epoch 30)" in content:
                    play_beeps()
                    break
            except Exception as e:
                print(f"Error reading file: {e}")
        
        time.sleep(15)  # Check every 15 seconds

if __name__ == "__main__":
    monitor()
