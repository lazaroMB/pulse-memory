package main

import (
	"context"
	"log"
	"os"

	"github.com/joho/godotenv"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
)

func main() {
	log.Println("Starting database cleanup...")

	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: No .env file found, using system environment variables.")
	}

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL is required in .env file.")
	}

	db, err := sqlx.Connect("pgx", dbURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Drop facts and relations tables
	log.Println("Dropping 'facts' and 'relations' tables (CASCADE)...")
	query := "DROP TABLE IF EXISTS facts, relations CASCADE;"
	
	_, err = db.ExecContext(ctx, query)
	if err != nil {
		log.Fatalf("Failed to drop tables: %v", err)
	}

	log.Println("Database cleanup completed successfully. Tables dropped.")
}
