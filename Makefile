.PHONY: all build run test test-client cleanup-db clean help

# Binary name and directory
BINARY_NAME=swarm-mem
BIN_DIR=bin

all: build

build:
	@echo "Compiling Go binaries..."
	@mkdir -p $(BIN_DIR)
	go build -o $(BIN_DIR)/$(BINARY_NAME) ./cmd/swarm-mem
	@echo "Build complete."

run: build
	@echo "Starting Swarm Memory Server..."
	./$(BIN_DIR)/$(BINARY_NAME)

test:
	@echo "Running unit tests..."
	go test -v ./...

test-client:
	@echo "Running integration test script..."
	./scripts/test_client.sh

cleanup-db:
	@echo "Executing database cleanup..."
	./scripts/cleanup_db.sh

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BIN_DIR)
	@echo "Clean complete."

help:
	@echo "Available commands in Swarm Memory Makefile:"
	@echo "  make build        - Compile the Go binary to $(BIN_DIR)/$(BINARY_NAME)"
	@echo "  make run          - Compile and run the HTTP server"
	@echo "  make test         - Execute unit tests"
	@echo "  make test-client  - Run the test_client.sh integration script"
	@echo "  make cleanup-db   - Erase facts and relations tables in database"
	@echo "  make clean        - Delete the compiled $(BIN_DIR) directory"
	@echo "  make help         - Display this help message"
