import re

def main():
    try:
        with open('research/results/determinism_check_run1.md', 'r') as f1:
            text1 = re.sub(r'\*Generated on: .*?\*', '', f1.read())
        with open('research/results/determinism_check_run2.md', 'r') as f2:
            text2 = re.sub(r'\*Generated on: .*?\*', '', f2.read())
            
        if text1 == text2:
            print("Identical ignoring timestamp")
            with open("research/results/determinism_check_final.md", "w") as out:
                out.write("Identical ignoring timestamp. Fixed source of randomness by injecting global seed into Python hash(), sklearn, and scipy RNGs.\n")
        else:
            print("Different")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
