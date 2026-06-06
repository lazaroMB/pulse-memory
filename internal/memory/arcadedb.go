package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"pulse/internal/document"
)

type ArcadeDBStore struct {
	url      string
	database string
	username string
	password string
	client   *http.Client
}

type CommandRequest struct {
	Language string                 `json:"language"`
	Command  string                 `json:"command"`
	Params   map[string]interface{} `json:"params,omitempty"`
}

func NewArcadeDBStore(url, database, username, password string) (*ArcadeDBStore, error) {
	if url == "" {
		url = "http://localhost:2480"
	}
	url = strings.TrimSuffix(url, "/")

	store := &ArcadeDBStore{
		url:      url,
		database: database,
		username: username,
		password: password,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	// Verify connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// ArcadeDB has a GET /api/v1/ready or similar check, or we can query DB ready.
	// We can execute a simple select query to check connection and verify database existence.
	_, err := store.execute(ctx, "sql", "SELECT 1", nil)
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "is not available") || strings.Contains(errStr, "not found") || strings.Contains(errStr, "500") {
			log.Printf("Database '%s' is not available. Attempting to create it automatically...", database)
			createErr := store.createDatabase(ctx)
			if createErr == nil {
				log.Printf("Database '%s' created successfully. Retrying connection check...", database)
				_, err = store.execute(ctx, "sql", "SELECT 1", nil)
			} else {
				return nil, fmt.Errorf("failed to verify connection to ArcadeDB: %w (and failed to auto-create: %v)", err, createErr)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("failed to verify connection to ArcadeDB: %w", err)
		}
	}

	return store, nil
}

func (s *ArcadeDBStore) createDatabase(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/v1/create/%s", s.url, s.database)
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.SetBasicAuth(s.username, s.password)

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute create database: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("create database failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func (s *ArcadeDBStore) Close() error {
	s.client.CloseIdleConnections()
	return nil
}

func (s *ArcadeDBStore) execute(ctx context.Context, language string, command string, params map[string]interface{}) ([]map[string]interface{}, error) {
	reqBody := CommandRequest{
		Language: language,
		Command:  command,
		Params:   params,
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/command/%s", s.url, s.database)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.SetBasicAuth(s.username, s.password)

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute command: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("command failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var respBody struct {
		Result json.RawMessage `json:"result"`
		Error  string          `json:"error"`
		Detail string          `json:"detail"`
	}
	if err := json.Unmarshal(bodyBytes, &respBody); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if respBody.Error != "" {
		return nil, fmt.Errorf("database error: %s - %s", respBody.Error, respBody.Detail)
	}

	var results []map[string]interface{}
	if len(respBody.Result) > 0 {
		firstChar := strings.TrimSpace(string(respBody.Result))
		if strings.HasPrefix(firstChar, "[") {
			if err := json.Unmarshal(respBody.Result, &results); err != nil {
				return nil, fmt.Errorf("failed to unmarshal array result: %w", err)
			}
		} else if strings.HasPrefix(firstChar, "{") {
			var singleMap map[string]interface{}
			if err := json.Unmarshal(respBody.Result, &singleMap); err != nil {
				return nil, fmt.Errorf("failed to unmarshal object result: %w", err)
			}
			results = []map[string]interface{}{singleMap}
		} else {
			var scalar interface{}
			_ = json.Unmarshal(respBody.Result, &scalar)
			results = []map[string]interface{}{{"value": scalar}}
		}
	}

	return results, nil
}

func (s *ArcadeDBStore) InitSchema(ctx context.Context) error {
	queries := []string{
		"CREATE VERTEX TYPE Entity IF NOT EXISTS",
		"CREATE PROPERTY Entity.id IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON Entity (id) UNIQUE",

		"CREATE VERTEX TYPE Fact IF NOT EXISTS",
		"CREATE PROPERTY Fact.id IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.entity_id IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.attribute IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.val IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.embedding IF NOT EXISTS LIST OF FLOAT",
		"CREATE PROPERTY Fact.confidence_score IF NOT EXISTS DOUBLE",
		"CREATE PROPERTY Fact.valid_from IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.valid_until IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.source_agent IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.memory_strength IF NOT EXISTS DOUBLE",
		"CREATE PROPERTY Fact.stability IF NOT EXISTS DOUBLE",
		"CREATE PROPERTY Fact.last_accessed IF NOT EXISTS STRING",
		"CREATE PROPERTY Fact.agent_owner IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON Fact (id) UNIQUE",
		"CREATE INDEX IF NOT EXISTS ON Fact (embedding) LSM_VECTOR METADATA { dimensions: 3072, similarity: 'COSINE' }",

		"CREATE VERTEX TYPE Document IF NOT EXISTS",
		"CREATE PROPERTY Document.id IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.title IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.source_type IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.source_url IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.file_path IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.status IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.error_message IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.metadata_json IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.created_at IF NOT EXISTS STRING",
		"CREATE PROPERTY Document.updated_at IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON Document (id) UNIQUE",

		"CREATE VERTEX TYPE DocumentChunk IF NOT EXISTS",
		"CREATE PROPERTY DocumentChunk.id IF NOT EXISTS STRING",
		"CREATE PROPERTY DocumentChunk.document_id IF NOT EXISTS STRING",
		"CREATE PROPERTY DocumentChunk.chunk_index IF NOT EXISTS INTEGER",
		"CREATE PROPERTY DocumentChunk.content IF NOT EXISTS STRING",
		"CREATE PROPERTY DocumentChunk.embedding IF NOT EXISTS LIST OF FLOAT",
		"CREATE PROPERTY DocumentChunk.metadata_json IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON DocumentChunk (id) UNIQUE",
		"CREATE INDEX IF NOT EXISTS ON DocumentChunk (embedding) LSM_VECTOR METADATA { dimensions: 3072, similarity: 'COSINE' }",

		"CREATE VERTEX TYPE CommunitySummary IF NOT EXISTS",
		"CREATE PROPERTY CommunitySummary.id IF NOT EXISTS STRING",
		"CREATE PROPERTY CommunitySummary.name IF NOT EXISTS STRING",
		"CREATE PROPERTY CommunitySummary.summary IF NOT EXISTS STRING",
		"CREATE PROPERTY CommunitySummary.embedding IF NOT EXISTS LIST OF FLOAT",
		"CREATE PROPERTY CommunitySummary.entities IF NOT EXISTS LIST OF STRING",
		"CREATE PROPERTY CommunitySummary.created_at IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON CommunitySummary (id) UNIQUE",
		"CREATE INDEX IF NOT EXISTS ON CommunitySummary (embedding) LSM_VECTOR METADATA { dimensions: 3072, similarity: 'COSINE' }",

		"CREATE EDGE TYPE HAS_FACT IF NOT EXISTS",
		
		"CREATE EDGE TYPE RELATION IF NOT EXISTS",
		"CREATE PROPERTY RELATION.id IF NOT EXISTS STRING",
		"CREATE PROPERTY RELATION.rel_type IF NOT EXISTS STRING",
		"CREATE PROPERTY RELATION.valid_from IF NOT EXISTS STRING",
		"CREATE PROPERTY RELATION.valid_until IF NOT EXISTS STRING",
		"CREATE PROPERTY RELATION.memory_strength IF NOT EXISTS DOUBLE",
		"CREATE PROPERTY RELATION.stability IF NOT EXISTS DOUBLE",
		"CREATE PROPERTY RELATION.last_accessed IF NOT EXISTS STRING",
		"CREATE PROPERTY RELATION.agent_owner IF NOT EXISTS STRING",
		"CREATE INDEX IF NOT EXISTS ON RELATION (id) UNIQUE",

		"CREATE EDGE TYPE AUTHORED IF NOT EXISTS",
		"CREATE EDGE TYPE DISCUSSES IF NOT EXISTS",
		"CREATE EDGE TYPE HAS_CHUNK IF NOT EXISTS",
		"CREATE EDGE TYPE EXTRACTED_FROM IF NOT EXISTS",
		"CREATE EDGE TYPE MENTIONED_IN IF NOT EXISTS",
	}

	for _, q := range queries {
		_, err := s.execute(ctx, "sql", q, nil)
		if err != nil {
			return fmt.Errorf("failed to execute schema initialization query [%s]: %w", q, err)
		}
	}

	return nil
}

func (s *ArcadeDBStore) InsertFact(ctx context.Context, fact *Fact, vector []float32) error {
	strength := fact.MemoryStrength
	if strength <= 0 {
		strength = 1.0
	}
	stability := fact.Stability
	if stability <= 0 && stability != -1.0 {
		stability = getDefaultStability()
	}
	lastAccessed := fact.LastAccessed
	if lastAccessed.IsZero() {
		lastAccessed = time.Now()
	}

	var validUntilStr interface{} = nil
	if fact.ValidUntil != nil {
		validUntilStr = fact.ValidUntil.Format(time.RFC3339Nano)
	}

	vecStr := float32SliceToSQLArray(vector)

	agentOwnerStr := ""
	if fact.AgentOwner != uuid.Nil {
		agentOwnerStr = fact.AgentOwner.String()
	} else if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	script := fmt.Sprintf(`BEGIN;
UPDATE Entity SET id = :entity_id UPSERT WHERE id = :entity_id;
CREATE VERTEX Fact SET id = :id, entity_id = :entity_id, attribute = :attribute, val = :val, embedding = %s, confidence_score = :confidence_score, valid_from = :valid_from, valid_until = :valid_until, source_agent = :source_agent, memory_strength = :memory_strength, stability = :stability, last_accessed = :last_accessed, agent_owner = :agent_owner;
CREATE EDGE HAS_FACT FROM (SELECT FROM Entity WHERE id = :entity_id) TO (SELECT FROM Fact WHERE id = :id) IF NOT EXISTS;
COMMIT;`, vecStr)

	params := map[string]interface{}{
		"entity_id":        fact.EntityID.String(),
		"id":              fact.ID.String(),
		"attribute":       fact.Attribute,
		"val":             fact.Value,
		"confidence_score": fact.ConfidenceScore,
		"valid_from":       fact.ValidFrom.Format(time.RFC3339Nano),
		"valid_until":      validUntilStr,
		"source_agent":     fact.SourceAgent,
		"memory_strength":  strength,
		"stability":        stability,
		"last_accessed":    lastAccessed.Format(time.RFC3339Nano),
		"agent_owner":      agentOwnerStr,
	}

	_, err := s.execute(ctx, "sqlscript", script, params)
	if err != nil {
		return fmt.Errorf("failed to insert fact: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) InsertFactWithProvenance(ctx context.Context, fact *Fact, vector []float32, docID uuid.UUID, chunkID uuid.UUID) error {
	if err := s.InsertFact(ctx, fact, vector); err != nil {
		return err
	}
	return s.LinkFactToSource(ctx, fact.ID, docID, chunkID)
}

func (s *ArcadeDBStore) SearchHybrid(ctx context.Context, q *MemorySearchQuery) ([]Fact, error) {
	var rows []map[string]interface{}
	var err error

	agentOwnerStr := ""
	if q.AgentOwner != uuid.Nil {
		agentOwnerStr = q.AgentOwner.String()
	} else if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	var ownerFilter string
	params := map[string]interface{}{
		"entity_id": q.TargetEntity.String(),
		"limit":     q.MaxResults,
	}
	if agentOwnerStr != "" {
		ownerFilter = " AND (agent_owner = :agent_owner OR agent_owner IS NULL OR agent_owner = '' OR agent_owner = :nil_owner)"
		params["agent_owner"] = agentOwnerStr
		params["nil_owner"] = uuid.Nil.String()
	}

	if len(q.QueryVector) == 0 {
		query := fmt.Sprintf(`SELECT FROM Fact WHERE entity_id = :entity_id%s AND (valid_until IS NULL OR valid_until = '') ORDER BY valid_from DESC LIMIT :limit`, ownerFilter)
		rows, err = s.execute(ctx, "sql", query, params)
	} else {
		vecStr := float32SliceToSQLArray(q.QueryVector)
		query := fmt.Sprintf(`SELECT FROM (
			SELECT expand(vectorNeighbors('Fact[embedding]', %s, 100))
		) WHERE entity_id = :entity_id%s AND (valid_until IS NULL OR valid_until = '') LIMIT :limit`, vecStr, ownerFilter)

		rows, err = s.execute(ctx, "sql", query, params)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to search facts: %w", err)
	}

	facts := make([]Fact, 0, len(rows))
	for _, row := range rows {
		idStr, _ := row["id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		entityStr, _ := row["entity_id"].(string)
		entityID, _ := uuid.Parse(entityStr)

		attr := ""
		if v, ok := row["attribute"].(string); ok {
			attr = v
		}
		val := ""
		if v, ok := row["val"].(string); ok {
			val = v
		}

		confidence := 0.0
		if cf, ok := row["confidence_score"].(float64); ok {
			confidence = cf
		} else if ci, ok := row["confidence_score"].(int64); ok {
			confidence = float64(ci)
		} else if ci, ok := row["confidence_score"].(int); ok {
			confidence = float64(ci)
		}

		validFromStr := ""
		if v, ok := row["valid_from"].(string); ok {
			validFromStr = v
		}
		validFrom, _ := time.Parse(time.RFC3339Nano, validFromStr)
		if validFrom.IsZero() {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr)
		}

		var validUntil *time.Time
		if vuStr, ok := row["valid_until"].(string); ok && vuStr != "" {
			t, err := time.Parse(time.RFC3339Nano, vuStr)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, vuStr)
			}
			if !t.IsZero() {
				validUntil = &t
			}
		}

		sourceAgent := ""
		if v, ok := row["source_agent"].(string); ok {
			sourceAgent = v
		}

		strength := 1.0
		if ms, ok := row["memory_strength"].(float64); ok {
			strength = ms
		}

		stability := 30.0
		if st, ok := row["stability"].(float64); ok {
			stability = st
		}

		var lastAccessed time.Time
		if laStr, ok := row["last_accessed"].(string); ok && laStr != "" {
			lastAccessed, _ = time.Parse(time.RFC3339Nano, laStr)
			if lastAccessed.IsZero() {
				lastAccessed, _ = time.Parse(time.RFC3339, laStr)
			}
		}

		agentOwner := uuid.Nil
		if v, ok := row["agent_owner"].(string); ok && v != "" {
			agentOwner, _ = uuid.Parse(v)
		}

		fact := Fact{
			ID:              id,
			EntityID:        entityID,
			Attribute:       attr,
			Value:           val,
			ConfidenceScore: confidence,
			ValidFrom:       validFrom,
			ValidUntil:      validUntil,
			SourceAgent:     sourceAgent,
			MemoryStrength:  strength,
			Stability:       stability,
			LastAccessed:    lastAccessed,
			AgentOwner:      agentOwner,
		}

		t := time.Since(fact.LastAccessed).Hours()
		stab := fact.Stability
		if stab == -1.0 {
			fact.MemoryStrength = 1.0
			facts = append(facts, fact)
			continue
		}
		if stab <= 0 {
			stab = getDefaultStability()
		}
		retention := math.Exp(-t / (stab * 24.0))
		if retention < 0.1 {
			go func(fid uuid.UUID) {
				_ = s.DeactivateFact(context.Background(), fid)
			}(fact.ID)
			continue
		}
		fact.MemoryStrength = retention
		facts = append(facts, fact)
	}

	for _, f := range facts {
		s.ReinforceFact(ctx, f.ID, f.Stability)
	}

	return facts, nil
}

func (s *ArcadeDBStore) ReinforceFact(ctx context.Context, factID uuid.UUID, currentStability float64) {
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if currentStability == -1.0 {
			query := "UPDATE Fact SET last_accessed = :last_accessed, memory_strength = 1.0 WHERE id = :id"
			_, _ = s.execute(bgCtx, "sql", query, map[string]interface{}{
				"last_accessed": time.Now().Format(time.RFC3339Nano),
				"id":            factID.String(),
			})
			return
		}

		newStability := currentStability * 1.5
		if newStability > 365.0 {
			newStability = 365.0
		}
		if newStability <= 0 {
			newStability = getDefaultStability() * 1.5
		}
		query := "UPDATE Fact SET last_accessed = :last_accessed, stability = :stability, memory_strength = 1.0 WHERE id = :id"
		_, _ = s.execute(bgCtx, "sql", query, map[string]interface{}{
			"last_accessed": time.Now().Format(time.RFC3339Nano),
			"stability":     newStability,
			"id":            factID.String(),
		})
	}()
}

func (s *ArcadeDBStore) DeactivateFact(ctx context.Context, factID uuid.UUID) error {
	query := "UPDATE Fact SET valid_until = :valid_until WHERE id = :id AND (valid_until IS NULL OR valid_until = '')"
	params := map[string]interface{}{
		"id":          factID.String(),
		"valid_until": time.Now().Format(time.RFC3339Nano),
	}
	_, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return fmt.Errorf("failed to deactivate fact: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) InsertRelation(ctx context.Context, relation *Relation) error {
	strength := relation.MemoryStrength
	if strength <= 0 {
		strength = 1.0
	}
	stability := relation.Stability
	if stability <= 0 {
		stability = getDefaultStability()
	}
	lastAccessed := relation.LastAccessed
	if lastAccessed.IsZero() {
		lastAccessed = time.Now()
	}

	var validUntilStr interface{} = nil
	if relation.ValidUntil != nil {
		validUntilStr = relation.ValidUntil.Format(time.RFC3339Nano)
	}

	agentOwnerStr := ""
	if relation.AgentOwner != uuid.Nil {
		agentOwnerStr = relation.AgentOwner.String()
	} else if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	script := `BEGIN;
UPDATE Entity SET id = :source_id UPSERT WHERE id = :source_id;
UPDATE Entity SET id = :target_id UPSERT WHERE id = :target_id;
CREATE EDGE RELATION FROM (SELECT FROM Entity WHERE id = :source_id) TO (SELECT FROM Entity WHERE id = :target_id)
  SET id = :id, rel_type = :rel_type, valid_from = :valid_from, valid_until = :valid_until, memory_strength = :memory_strength, stability = :stability, last_accessed = :last_accessed, agent_owner = :agent_owner;
COMMIT;`

	params := map[string]interface{}{
		"source_id":       relation.SourceID.String(),
		"target_id":       relation.TargetID.String(),
		"id":              relation.ID.String(),
		"rel_type":        relation.Type,
		"valid_from":      relation.ValidFrom.Format(time.RFC3339Nano),
		"valid_until":     validUntilStr,
		"memory_strength": strength,
		"stability":       stability,
		"last_accessed":   lastAccessed.Format(time.RFC3339Nano),
		"agent_owner":      agentOwnerStr,
	}

	_, err := s.execute(ctx, "sqlscript", script, params)
	if err != nil {
		return fmt.Errorf("failed to insert relation: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error) {
	agentOwnerStr := ""
	if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	var ownerFilter string
	params := map[string]interface{}{
		"entity_id": entityID.String(),
	}
	if agentOwnerStr != "" {
		ownerFilter = " AND (agent_owner = :agent_owner OR agent_owner IS NULL OR agent_owner = '' OR agent_owner = :nil_owner)"
		params["agent_owner"] = agentOwnerStr
		params["nil_owner"] = uuid.Nil.String()
	}

	query := fmt.Sprintf(`SELECT id, @out.id AS source_id, @in.id AS target_id, rel_type, valid_from, valid_until, memory_strength, stability, last_accessed, agent_owner 
		FROM RELATION 
		WHERE (@out.id = :entity_id OR @in.id = :entity_id)%s AND (valid_until IS NULL OR valid_until = '')`, ownerFilter)

	rows, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations: %w", err)
	}

	relations := make([]Relation, 0, len(rows))
	for _, row := range rows {
		idStr, _ := row["id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		srcStr, _ := row["source_id"].(string)
		srcID, err := uuid.Parse(srcStr)
		if err != nil {
			continue
		}

		tgtStr, _ := row["target_id"].(string)
		tgtID, err := uuid.Parse(tgtStr)
		if err != nil {
			continue
		}

		relType := ""
		if v, ok := row["rel_type"].(string); ok {
			relType = v
		}

		validFromStr := ""
		if v, ok := row["valid_from"].(string); ok {
			validFromStr = v
		}
		validFrom, _ := time.Parse(time.RFC3339Nano, validFromStr)
		if validFrom.IsZero() {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr)
		}

		var validUntil *time.Time
		if vuStr, ok := row["valid_until"].(string); ok && vuStr != "" {
			t, err := time.Parse(time.RFC3339Nano, vuStr)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, vuStr)
			}
			if !t.IsZero() {
				validUntil = &t
			}
		}

		strength := 1.0
		if ms, ok := row["memory_strength"].(float64); ok {
			strength = ms
		}
		stability := 30.0
		if st, ok := row["stability"].(float64); ok {
			stability = st
		}
		var lastAccessed time.Time
		if laStr, ok := row["last_accessed"].(string); ok && laStr != "" {
			lastAccessed, _ = time.Parse(time.RFC3339Nano, laStr)
			if lastAccessed.IsZero() {
				lastAccessed, _ = time.Parse(time.RFC3339, laStr)
			}
		}

		agentOwner := uuid.Nil
		if v, ok := row["agent_owner"].(string); ok && v != "" {
			agentOwner, _ = uuid.Parse(v)
		}

		relations = append(relations, Relation{
			ID:             id,
			SourceID:       srcID,
			TargetID:       tgtID,
			Type:           relType,
			ValidFrom:      validFrom,
			ValidUntil:     validUntil,
			MemoryStrength: strength,
			Stability:      stability,
			LastAccessed:   lastAccessed,
			AgentOwner:      agentOwner,
		})
	}

	return relations, nil
}

func (s *ArcadeDBStore) InsertDocument(ctx context.Context, doc *document.Document) error {
	metadataBytes, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal document metadata: %w", err)
	}

	query := `CREATE VERTEX Document SET id = :id, title = :title, source_type = :source_type, source_url = :source_url, file_path = :file_path, status = :status, error_message = :error_message, metadata_json = :metadata_json, created_at = :created_at, updated_at = :updated_at`

	params := map[string]interface{}{
		"id":            doc.ID.String(),
		"title":         doc.Title,
		"source_type":   string(doc.SourceType),
		"source_url":    doc.SourceURL,
		"file_path":     doc.FilePath,
		"status":        string(doc.Status),
		"error_message": doc.ErrorMessage,
		"metadata_json": string(metadataBytes),
		"created_at":    doc.CreatedAt.Format(time.RFC3339Nano),
		"updated_at":    doc.UpdatedAt.Format(time.RFC3339Nano),
	}

	_, err = s.execute(ctx, "sql", query, params)
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) GetDocument(ctx context.Context, id uuid.UUID) (*document.Document, error) {
	rows, err := s.execute(ctx, "sql", "SELECT FROM Document WHERE id = :id", map[string]interface{}{
		"id": id.String(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get document: %w", err)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("document not found")
	}

	row := rows[0]

	title := ""
	if v, ok := row["title"].(string); ok {
		title = v
	}
	srcType := ""
	if v, ok := row["source_type"].(string); ok {
		srcType = v
	}
	srcURL := ""
	if v, ok := row["source_url"].(string); ok {
		srcURL = v
	}
	filePath := ""
	if v, ok := row["file_path"].(string); ok {
		filePath = v
	}
	status := ""
	if v, ok := row["status"].(string); ok {
		status = v
	}
	errMsg := ""
	if v, ok := row["error_message"].(string); ok {
		errMsg = v
	}

	var metadata map[string]string
	if metaStr, ok := row["metadata_json"].(string); ok && metaStr != "" {
		_ = json.Unmarshal([]byte(metaStr), &metadata)
	}

	createdStr := ""
	if v, ok := row["created_at"].(string); ok {
		createdStr = v
	}
	updatedStr := ""
	if v, ok := row["updated_at"].(string); ok {
		updatedStr = v
	}

	createdAt, _ := time.Parse(time.RFC3339Nano, createdStr)
	if createdAt.IsZero() {
		createdAt, _ = time.Parse(time.RFC3339, createdStr)
	}
	updatedAt, _ := time.Parse(time.RFC3339Nano, updatedStr)
	if updatedAt.IsZero() {
		updatedAt, _ = time.Parse(time.RFC3339, updatedStr)
	}

	return &document.Document{
		ID:           id,
		Title:        title,
		SourceType:   document.SourceType(srcType),
		SourceURL:    srcURL,
		FilePath:     filePath,
		Status:       document.IngestionStatus(status),
		ErrorMessage: errMsg,
		Metadata:     metadata,
		CreatedAt:    createdAt,
		UpdatedAt:    updatedAt,
	}, nil
}

func (s *ArcadeDBStore) UpdateDocumentStatus(ctx context.Context, docID uuid.UUID, status document.IngestionStatus, errMsg string) error {
	query := "UPDATE Document SET status = :status, error_message = :error_message, updated_at = :updated_at WHERE id = :id"
	params := map[string]interface{}{
		"id":            docID.String(),
		"status":        string(status),
		"error_message": errMsg,
		"updated_at":    time.Now().Format(time.RFC3339Nano),
	}
	_, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return fmt.Errorf("failed to update document status: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) InsertDocumentChunks(ctx context.Context, chunks []document.DocumentChunk, embeddings [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}

	var sb strings.Builder
	sb.WriteString("BEGIN;\n")
	params := make(map[string]interface{})

	for i, chunk := range chunks {
		metadataBytes, _ := json.Marshal(chunk.Metadata)
		vecStr := "[]"
		if len(embeddings) > i && len(embeddings[i]) > 0 {
			vecStr = float32SliceToSQLArray(embeddings[i])
		}

		chunkKey := fmt.Sprintf("chunk_%d", i)
		sb.WriteString(fmt.Sprintf(`CREATE VERTEX DocumentChunk SET id = :id_%[1]s, document_id = :document_id_%[1]s, chunk_index = :chunk_index_%[1]s, content = :content_%[1]s, embedding = %[2]s, metadata_json = :metadata_json_%[1]s;
CREATE EDGE HAS_CHUNK FROM (SELECT FROM Document WHERE id = :document_id_%[1]s) TO (SELECT FROM DocumentChunk WHERE id = :id_%[1]s) IF NOT EXISTS;
`, chunkKey, vecStr))

		params["id_"+chunkKey] = chunk.ID.String()
		params["document_id_"+chunkKey] = chunk.DocumentID.String()
		params["chunk_index_"+chunkKey] = chunk.ChunkIndex
		params["content_"+chunkKey] = chunk.Content
		params["metadata_json_"+chunkKey] = string(metadataBytes)
	}
	sb.WriteString("COMMIT;")

	_, err := s.execute(ctx, "sqlscript", sb.String(), params)
	if err != nil {
		return fmt.Errorf("failed to insert document chunks: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]document.DocumentChunk, error) {
	if len(queryVector) == 0 {
		return nil, nil
	}

	vecStr := float32SliceToSQLArray(queryVector)
	query := fmt.Sprintf("SELECT expand(vectorNeighbors('DocumentChunk[embedding]', %s, :limit))", vecStr)
	rows, err := s.execute(ctx, "sql", query, map[string]interface{}{
		"limit":        limit,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search document chunks: %w", err)
	}

	chunks := make([]document.DocumentChunk, 0, len(rows))
	for _, row := range rows {
		idStr, _ := row["id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		var docID uuid.UUID
		if docIDStr, ok := row["document_id"].(string); ok {
			docID, _ = uuid.Parse(docIDStr)
		}

		var chunkIdx int
		if v, ok := row["chunk_index"].(float64); ok {
			chunkIdx = int(v)
		} else if v, ok := row["chunk_index"].(int); ok {
			chunkIdx = v
		} else if v, ok := row["chunk_index"].(int64); ok {
			chunkIdx = int(v)
		}

		content := ""
		if v, ok := row["content"].(string); ok {
			content = v
		}

		var metadata map[string]string
		if metaStr, ok := row["metadata_json"].(string); ok && metaStr != "" {
			_ = json.Unmarshal([]byte(metaStr), &metadata)
		}

		chunks = append(chunks, document.DocumentChunk{
			ID:         id,
			DocumentID: docID,
			ChunkIndex: chunkIdx,
			Content:    content,
			Metadata:   metadata,
		})
	}
	return chunks, nil
}

func (s *ArcadeDBStore) LinkDocumentToAuthor(ctx context.Context, docID uuid.UUID, authorID uuid.UUID) error {
	script := `BEGIN;
UPDATE Document SET id = :docID UPSERT WHERE id = :docID;
UPDATE Entity SET id = :authorID UPSERT WHERE id = :authorID;
CREATE EDGE AUTHORED FROM (SELECT FROM Entity WHERE id = :authorID) TO (SELECT FROM Document WHERE id = :docID) IF NOT EXISTS;
COMMIT;`
	_, err := s.execute(ctx, "sqlscript", script, map[string]interface{}{
		"docID":    docID.String(),
		"authorID": authorID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to link document to author: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) LinkDocumentToTopic(ctx context.Context, docID uuid.UUID, topicName string) error {
	topicID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("topic-"+topicName))
	script := `BEGIN;
UPDATE Document SET id = :docID UPSERT WHERE id = :docID;
UPDATE Entity SET id = :topicID, attribute = 'topic', val = :topicName UPSERT WHERE id = :topicID;
CREATE EDGE DISCUSSES FROM (SELECT FROM Document WHERE id = :docID) TO (SELECT FROM Entity WHERE id = :topicID) IF NOT EXISTS;
COMMIT;`
	_, err := s.execute(ctx, "sqlscript", script, map[string]interface{}{
		"docID":     docID.String(),
		"topicID":   topicID.String(),
		"topicName": topicName,
	})
	if err != nil {
		return fmt.Errorf("failed to link document to topic: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) LinkFactToSource(ctx context.Context, factID uuid.UUID, docID uuid.UUID, chunkID uuid.UUID) error {
	var script string
	params := map[string]interface{}{
		"factID": factID.String(),
		"docID":  docID.String(),
	}

	if chunkID != uuid.Nil {
		script = `BEGIN;
CREATE EDGE EXTRACTED_FROM FROM (SELECT FROM Fact WHERE id = :factID) TO (SELECT FROM Document WHERE id = :docID) IF NOT EXISTS;
CREATE EDGE MENTIONED_IN FROM (SELECT FROM Fact WHERE id = :factID) TO (SELECT FROM DocumentChunk WHERE id = :chunkID) IF NOT EXISTS;
COMMIT;`
		params["chunkID"] = chunkID.String()
	} else {
		script = `BEGIN;
CREATE EDGE EXTRACTED_FROM FROM (SELECT FROM Fact WHERE id = :factID) TO (SELECT FROM Document WHERE id = :docID) IF NOT EXISTS;
COMMIT;`
	}

	_, err := s.execute(ctx, "sqlscript", script, params)
	if err != nil {
		return fmt.Errorf("failed to link fact to source: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) GetAllEntities(ctx context.Context) ([]EntityNode, error) {
	agentOwnerStr := ""
	if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	var ownerFilter string
	var params map[string]interface{}
	if agentOwnerStr != "" {
		ownerFilter = " AND (agent_owner = :agent_owner OR agent_owner IS NULL OR agent_owner = '' OR agent_owner = :nil_owner)"
		params = map[string]interface{}{
			"agent_owner": agentOwnerStr,
			"nil_owner":   uuid.Nil.String(),
		}
	}

	query := fmt.Sprintf("SELECT entity_id, val, embedding FROM Fact WHERE attribute = 'name'%s AND (valid_until IS NULL OR valid_until = '')", ownerFilter)
	rows, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to select entities: %w", err)
	}

	entities := make([]EntityNode, 0, len(rows))
	for _, r := range rows {
		idStr, _ := r["entity_id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		name, _ := r["val"].(string)

		var embedding []float32
		if embList, ok := r["embedding"].([]interface{}); ok {
			embedding = make([]float32, len(embList))
			for i, v := range embList {
				if fv, ok := v.(float64); ok {
					embedding[i] = float32(fv)
				}
			}
		}

		entities = append(entities, EntityNode{
			ID:        id,
			Name:      name,
			Embedding: embedding,
		})
	}
	return entities, nil
}

func (s *ArcadeDBStore) MergeEntities(ctx context.Context, canonicalID, duplicateID uuid.UUID) error {
	_, err := s.execute(ctx, "sql", "UPDATE Fact SET entity_id = :canonicalID WHERE entity_id = :duplicateID", map[string]interface{}{
		"canonicalID": canonicalID.String(),
		"duplicateID": duplicateID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to migrate fact entity_ids: %w", err)
	}

	_, err = s.execute(ctx, "sql", "CREATE EDGE HAS_FACT FROM (SELECT FROM Entity WHERE id = :canonicalID) TO (SELECT FROM Fact WHERE entity_id = :canonicalID) IF NOT EXISTS", map[string]interface{}{
		"canonicalID": canonicalID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to recreate HAS_FACT edges: %w", err)
	}

	relRows, err := s.execute(ctx, "sql", "SELECT *, @out.id AS out_id, @in.id AS in_id, agent_owner FROM RELATION WHERE @out.id = :duplicateID OR @in.id = :duplicateID", map[string]interface{}{
		"duplicateID": duplicateID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to fetch duplicate relations: %w", err)
	}

	for _, row := range relRows {
		idStr, _ := row["id"].(string)
		relType, _ := row["rel_type"].(string)
		validFrom, _ := row["valid_from"].(string)
		var validUntil interface{} = nil
		if vu, ok := row["valid_until"]; ok && vu != nil {
			validUntil = vu
		}

		memStrength := 1.0
		if ms, ok := row["memory_strength"].(float64); ok {
			memStrength = ms
		}
		stability := 30.0
		if st, ok := row["stability"].(float64); ok {
			stability = st
		}
		lastAccessed, _ := row["last_accessed"].(string)

		agentOwnerStr := ""
		if ao, ok := row["agent_owner"].(string); ok {
			agentOwnerStr = ao
		}

		outID, _ := row["out_id"].(string)
		inID, _ := row["in_id"].(string)

		newOutID := outID
		newInID := inID
		if outID == duplicateID.String() {
			newOutID = canonicalID.String()
		}
		if inID == duplicateID.String() {
			newInID = canonicalID.String()
		}

		createRelQuery := `CREATE EDGE RELATION FROM (SELECT FROM Entity WHERE id = :outID) TO (SELECT FROM Entity WHERE id = :inID)
			SET id = :id, rel_type = :rel_type, valid_from = :valid_from, valid_until = :valid_until,
			    memory_strength = :memory_strength, stability = :stability, last_accessed = :last_accessed, agent_owner = :agent_owner`

		_, err = s.execute(ctx, "sql", createRelQuery, map[string]interface{}{
			"outID":           newOutID,
			"inID":            newInID,
			"id":              idStr,
			"rel_type":        relType,
			"valid_from":      validFrom,
			"valid_until":     validUntil,
			"memory_strength": memStrength,
			"stability":       stability,
			"last_accessed":   lastAccessed,
			"agent_owner":     agentOwnerStr,
		})
		if err != nil {
			log.Printf("Warning: failed to migrate RELATION edge %s: %v", idStr, err)
		}
	}

	_, err = s.execute(ctx, "sql", "DELETE VERTEX Entity WHERE id = :duplicateID", map[string]interface{}{
		"duplicateID": duplicateID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to delete duplicate entity: %w", err)
	}

	nameFacts, err := s.execute(ctx, "sql", "SELECT id FROM Fact WHERE entity_id = :canonicalID AND attribute = 'name' AND (valid_until IS NULL OR valid_until = '') ORDER BY valid_from DESC", map[string]interface{}{
		"canonicalID": canonicalID.String(),
	})
	if err != nil {
		return fmt.Errorf("failed to fetch canonical name facts: %w", err)
	}

	if len(nameFacts) > 1 {
		newestFactID := nameFacts[0]["id"].(string)
		for i := 1; i < len(nameFacts); i++ {
			factID, _ := nameFacts[i]["id"].(string)
			if factID == newestFactID {
				continue
			}
			_, err = s.execute(ctx, "sql", "UPDATE Fact SET valid_until = :valid_until WHERE id = :factID", map[string]interface{}{
				"valid_until": time.Now().Format(time.RFC3339Nano),
				"factID":      factID,
			})
			if err != nil {
				log.Printf("Warning: failed to deactivate duplicate name fact %s: %v", factID, err)
			}
		}
	}

	return nil
}

func (s *ArcadeDBStore) InsertCommunitySummary(ctx context.Context, id uuid.UUID, name string, summary string, embedding []float32, entities []uuid.UUID) error {
	vecStr := float32SliceToSQLArray(embedding)
	entStr := uuidSliceToSQLArray(entities)

	query := fmt.Sprintf(`UPDATE CommunitySummary SET id = :id, name = :name, summary = :summary, embedding = %s, entities = %s UPSERT WHERE id = :id`, vecStr, entStr)
	params := map[string]interface{}{
		"id":      id.String(),
		"name":    name,
		"summary": summary,
	}

	_, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return fmt.Errorf("failed to insert community summary: %w", err)
	}
	return nil
}

func (s *ArcadeDBStore) SearchCommunitySummaries(ctx context.Context, queryVector []float32, limit int) ([]CommunitySummary, error) {
	if len(queryVector) == 0 {
		return nil, nil
	}

	vecStr := float32SliceToSQLArray(queryVector)
	query := fmt.Sprintf("SELECT expand(vectorNeighbors('CommunitySummary[embedding]', %s, :limit))", vecStr)
	rows, err := s.execute(ctx, "sql", query, map[string]interface{}{
		"limit":        limit,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search community summaries: %w", err)
	}

	summaries := make([]CommunitySummary, 0, len(rows))
	for _, row := range rows {
		idStr, _ := row["id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		name := ""
		if v, ok := row["name"].(string); ok {
			name = v
		}
		summary := ""
		if v, ok := row["summary"].(string); ok {
			summary = v
		}

		var entities []uuid.UUID
		if entList, ok := row["entities"].([]interface{}); ok {
			entities = make([]uuid.UUID, 0, len(entList))
			for _, item := range entList {
				if itemStr, ok := item.(string); ok {
					if u, err := uuid.Parse(itemStr); err == nil {
						entities = append(entities, u)
					}
				}
			}
		}

		var createdAt time.Time
		if createdStr, ok := row["created_at"].(string); ok {
			createdAt, _ = time.Parse(time.RFC3339Nano, createdStr)
			if createdAt.IsZero() {
				createdAt, _ = time.Parse(time.RFC3339, createdStr)
			}
		}

		summaries = append(summaries, CommunitySummary{
			ID:        id,
			Name:      name,
			Summary:   summary,
			Entities:  entities,
			CreatedAt: createdAt,
		})
	}
	return summaries, nil
}

func (s *ArcadeDBStore) GetActiveRelationsBatch(ctx context.Context, entityIDs []uuid.UUID) ([]Relation, error) {
	if len(entityIDs) == 0 {
		return nil, nil
	}

	agentOwnerStr := ""
	if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	entityStrs := make([]interface{}, len(entityIDs))
	for i, ent := range entityIDs {
		entityStrs[i] = ent.String()
	}

	var ownerFilter string
	params := map[string]interface{}{
		"entity_ids": entityStrs,
	}
	if agentOwnerStr != "" {
		ownerFilter = " AND (agent_owner = :agent_owner OR agent_owner IS NULL OR agent_owner = '' OR agent_owner = :nil_owner)"
		params["agent_owner"] = agentOwnerStr
		params["nil_owner"] = uuid.Nil.String()
	}

	query := fmt.Sprintf(`SELECT id, @out.id AS source_id, @in.id AS target_id, rel_type, valid_from, valid_until, memory_strength, stability, last_accessed, agent_owner 
		FROM RELATION 
		WHERE (@out.id IN :entity_ids OR @in.id IN :entity_ids)%s AND (valid_until IS NULL OR valid_until = '')`, ownerFilter)

	rows, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations batch: %w", err)
	}

	relations := make([]Relation, 0, len(rows))
	for _, row := range rows {
		idStr, _ := row["id"].(string)
		id, err := uuid.Parse(idStr)
		if err != nil {
			continue
		}

		srcStr, _ := row["source_id"].(string)
		srcID, err := uuid.Parse(srcStr)
		if err != nil {
			continue
		}

		tgtStr, _ := row["target_id"].(string)
		tgtID, err := uuid.Parse(tgtStr)
		if err != nil {
			continue
		}

		relType := ""
		if v, ok := row["rel_type"].(string); ok {
			relType = v
		}

		validFromStr := ""
		if v, ok := row["valid_from"].(string); ok {
			validFromStr = v
		}
		validFrom, _ := time.Parse(time.RFC3339Nano, validFromStr)
		if validFrom.IsZero() {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr)
		}

		var validUntil *time.Time
		if vuStr, ok := row["valid_until"].(string); ok && vuStr != "" {
			t, err := time.Parse(time.RFC3339Nano, vuStr)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, vuStr)
			}
			if !t.IsZero() {
				validUntil = &t
			}
		}

		strength := 1.0
		if ms, ok := row["memory_strength"].(float64); ok {
			strength = ms
		}
		stability := 30.0
		if st, ok := row["stability"].(float64); ok {
			stability = st
		}
		var lastAccessed time.Time
		if laStr, ok := row["last_accessed"].(string); ok && laStr != "" {
			lastAccessed, _ = time.Parse(time.RFC3339Nano, laStr)
			if lastAccessed.IsZero() {
				lastAccessed, _ = time.Parse(time.RFC3339, laStr)
			}
		}

		agentOwner := uuid.Nil
		if v, ok := row["agent_owner"].(string); ok && v != "" {
			agentOwner, _ = uuid.Parse(v)
		}

		relations = append(relations, Relation{
			ID:             id,
			SourceID:       srcID,
			TargetID:       tgtID,
			Type:           relType,
			ValidFrom:      validFrom,
			ValidUntil:     validUntil,
			MemoryStrength: strength,
			Stability:      stability,
			LastAccessed:   lastAccessed,
			AgentOwner:      agentOwner,
		})
	}

	return relations, nil
}

func (s *ArcadeDBStore) GetEntityNamesBatch(ctx context.Context, entityIDs []uuid.UUID) (map[uuid.UUID]string, error) {
	if len(entityIDs) == 0 {
		return nil, nil
	}

	agentOwnerStr := ""
	if ownerID, ok := GetAgentOwner(ctx); ok {
		agentOwnerStr = ownerID.String()
	}

	entityStrs := make([]interface{}, len(entityIDs))
	for i, ent := range entityIDs {
		entityStrs[i] = ent.String()
	}

	var ownerFilter string
	params := map[string]interface{}{
		"entity_ids": entityStrs,
	}
	if agentOwnerStr != "" {
		ownerFilter = " AND (agent_owner = :agent_owner OR agent_owner IS NULL OR agent_owner = '' OR agent_owner = :nil_owner)"
		params["agent_owner"] = agentOwnerStr
		params["nil_owner"] = uuid.Nil.String()
	}

	query := fmt.Sprintf("SELECT entity_id, val FROM Fact WHERE entity_id IN :entity_ids AND attribute = 'name'%s AND (valid_until IS NULL OR valid_until = '')", ownerFilter)
	rows, err := s.execute(ctx, "sql", query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get entity names batch: %w", err)
	}

	nameMap := make(map[uuid.UUID]string)
	for _, row := range rows {
		entStr, _ := row["entity_id"].(string)
		entID, err := uuid.Parse(entStr)
		if err != nil {
			continue
		}
		name, _ := row["val"].(string)
		nameMap[entID] = name
	}

	return nameMap, nil
}

func getDefaultStability() float64 {
	val := os.Getenv("DEFAULT_MEMORY_STABILITY_DAYS")
	if val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 30.0
}

func float32SliceToSQLArray(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, val := range v {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.FormatFloat(float64(val), 'f', 6, 32))
	}
	sb.WriteByte(']')
	return sb.String()
}

func uuidSliceToSQLArray(uuids []uuid.UUID) string {
	if len(uuids) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, u := range uuids {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteByte('\'')
		sb.WriteString(u.String())
		sb.WriteByte('\'')
	}
	sb.WriteByte(']')
	return sb.String()
}
