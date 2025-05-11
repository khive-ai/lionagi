#!/bin/bash

# pre-commit-run-lionag2.sh - Script to manage pre-commit hooks for lionag2 project
# Default behavior: Run pre-commit checks on all files in the lionag2 folder

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Navigate to the lionag2 folder
cd "$(dirname "$0")/../lion/lionag2" || {
  echo -e "${RED}Error: Could not navigate to lionag2 folder${NC}"
  exit 1
}

# Check if pre-commit is installed
if ! command_exists pre-commit; then
  echo -e "${YELLOW}pre-commit is not installed. Installing...${NC}"
  
  # Try uv installation
  if command_exists uv; then
    uv pip install pre-commit
  elif command_exists pip3; then
    pip3 install pre-commit
  elif command_exists pip; then
    pip install pre-commit
  else
    echo -e "${RED}Error: uv, pip or pip3 not found. Please install Python and uv first.${NC}"
    echo "You can install pre-commit manually with: uv pip install pre-commit"
    exit 1
  fi
  
  echo -e "${GREEN}pre-commit installed successfully!${NC}"
else
  echo -e "${GREEN}pre-commit is already installed.${NC}"
fi

# Function to install hooks
install_hooks() {
  echo -e "${BLUE}Installing pre-commit hooks for lionag2...${NC}"
  pre-commit install
  echo -e "${GREEN}Hooks installed successfully!${NC}"
}

# Function to run hooks on all files
run_all() {
  echo -e "${BLUE}Running pre-commit hooks on all files in lionag2...${NC}"
  pre-commit run --all-files
}

# Function to run hooks on staged files
run_staged() {
  echo -e "${BLUE}Running pre-commit hooks on staged files in lionag2...${NC}"
  pre-commit run
}

# Function to update hooks
update_hooks() {
  echo -e "${BLUE}Updating pre-commit hooks for lionag2...${NC}"
  pre-commit autoupdate
  echo -e "${GREEN}Hooks updated successfully!${NC}"
}

# Function to show interactive menu
show_menu() {
  echo -e "\n${BLUE}=== Pre-commit Hook Manager for lionag2 ===${NC}"
  echo "1. Install hooks"
  echo "2. Run hooks on all files"
  echo "3. Run hooks on staged files"
  echo "4. Update hooks"
  echo "q. Quit"
  
  read -p "Enter your choice: " choice
  
  case "$choice" in
    1) install_hooks ;;
    2) run_all ;;
    3) run_staged ;;
    4) update_hooks ;;
    q|Q) echo -e "${GREEN}Exiting.${NC}" && exit 0 ;;
    *) echo -e "${RED}Invalid option.${NC}" && exit 1 ;;
  esac
}

# Process command line arguments
case "${1:-lint}" in
  install)
    install_hooks
    ;;
  lint|run-all)
    run_all
    ;;
  staged|run)
    run_staged
    ;;
  update)
    update_hooks
    ;;
  menu)
    show_menu
    ;;
  --help|-h) 
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./$(basename "$0") [command]"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "  lint       Run hooks on all files (default)"
    echo "  staged     Run hooks on staged files only"
    echo "  install    Install pre-commit hooks"
    echo "  update     Update hooks to latest versions"
    echo "  menu       Show interactive menu"
    ;;
  *)
    echo -e "${RED}Unknown command: $1${NC}"
    echo "Use --help for usage information."
    run_all  # Default to running all if unknown command
    ;;
esac