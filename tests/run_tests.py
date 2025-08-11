#!/usr/bin/env python
"""
Test runner script for astralora project.

This script provides convenient commands for running different types of tests.
"""

import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run astralora tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--bb-layers", action="store_true", help="Run BB layer tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip slow)")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.quick:
        base_cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        base_cmd.extend(["--cov=core", "--cov-report=term"])
        # Note: Add --cov-report=html if you want HTML coverage reports
    
    exit_codes = []
    
    if args.all:
        # Run all tests
        cmd = base_cmd + ["tests/"]
        exit_codes.append(run_command(cmd, "All Tests"))
    
    elif args.bb_layers:
        # Run BB layer tests
        cmd = base_cmd + ["tests/bb_layers/"]
        exit_codes.append(run_command(cmd, "BB Layer Tests"))
    
    else:
        # Default: run BB layer tests
        cmd = base_cmd + ["tests/bb_layers/"]
        exit_codes.append(run_command(cmd, "BB Layer Tests (default)"))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary:")
    if all(code == 0 for code in exit_codes):
        print("🎉 All tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()