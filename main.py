"""
Main entry point for CloneWiper (Qt UI).
"""
import sys
import os

def main():
    """Main entry point for CloneWiper (Qt UI)."""
    try:
        from qt_app import main as launch_qt
        sys.exit(launch_qt())
    except ImportError as e:
        print(f"Error: Required dependencies not found. {e}")
        print("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



