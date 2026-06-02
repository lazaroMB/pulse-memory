package consolidation

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"

	"github.com/google/uuid"
	"pulse/internal/memory"
)

// RunEntityResolution recorre todas las entidades activas, calcula similitudes lingüísticas y semánticas, y fusiona duplicados.
func RunEntityResolution(ctx context.Context, store memory.MemoryStore) error {
	entities, err := store.GetAllEntities(ctx)
	if err != nil {
		return fmt.Errorf("failed to retrieve entities: %w", err)
	}

	if len(entities) < 2 {
		return nil
	}

	log.Printf("[Entity Resolution] Running entity resolution scan across %d active entities...", len(entities))
	mergedCount := 0

	// Usar un mapa para llevar registro de las entidades que ya han sido fusionadas en este ciclo
	merged := make(map[uuid.UUID]uuid.UUID) // duplicateID -> canonicalID

	// Comparación cuadrática N^2
	for i := 0; i < len(entities); i++ {
		entA := entities[i]

		// Si entA ya fue marcado como duplicado y fusionado en otra entidad, omitir
		if _, isDup := merged[entA.ID]; isDup {
			continue
		}

		for j := i + 1; j < len(entities); j++ {
			entB := entities[j]

			// Si entB ya fue fusionado o es la misma entidad
			if _, isDup := merged[entB.ID]; isDup || entA.ID == entB.ID {
				continue
			}

			// 1. Similitud Semántica (Cosine Similarity de embeddings de nombres)
			var vecSim float64
			if len(entA.Embedding) > 0 && len(entB.Embedding) > 0 {
				var err error
				vecSim, err = memory.CosineSimilarity(entA.Embedding, entB.Embedding)
				if err != nil {
					vecSim = 0.0
				}
			}

			// 2. Similitud Lingüística (Jaro-Winkler)
			strSim := CalculateJaroWinkler(entA.Name, entB.Name)

			// 3. Similitud Híbrida Ponderada (60% semántica, 40% tipográfica)
			hibridSim := 0.6*vecSim + 0.4*strSim

			// Umbral de resolución de entidades >= 0.93
			if hibridSim >= 0.93 {
				// Elegir entidad canónica deterministicamente basado en comparación lexicográfica del UUID
				canonical := entA
				duplicate := entB
				if entB.ID.String() < entA.ID.String() {
					canonical = entB
					duplicate = entA
				}

				log.Printf("[Entity Resolution] Match detected: '%s' (%s) <-> '%s' (%s) [Semantic: %.2f, Text: %.2f, Combined: %.2f]. Merging...",
					canonical.Name, canonical.ID, duplicate.Name, duplicate.ID, vecSim, strSim, hibridSim)

				err := store.MergeEntities(ctx, canonical.ID, duplicate.ID)
				if err != nil {
					log.Printf("[Entity Resolution] Error merging entity %s into %s: %v", duplicate.ID, canonical.ID, err)
					continue
				}

				merged[duplicate.ID] = canonical.ID
				mergedCount++

				// Si la entidad A fue fusionada, salir del bucle interno
				if duplicate.ID == entA.ID {
					break
				}
			}
		}
	}

	if mergedCount > 0 {
		log.Printf("[Entity Resolution] Scan completed. Successfully merged %d duplicate entities.", mergedCount)
	} else {
		log.Printf("[Entity Resolution] Scan completed. No duplicates detected above threshold 0.93.")
	}

	return nil
}

// CalculateJaroWinkler calcula la distancia de Jaro-Winkler entre s1 y s2 para absorber errores y variaciones tipográficas.
func CalculateJaroWinkler(s1, s2 string) float64 {
	s1 = strings.TrimSpace(strings.ToLower(s1))
	s2 = strings.TrimSpace(strings.ToLower(s2))

	if s1 == s2 {
		return 1.0
	}

	len1 := len(s1)
	len2 := len(s2)

	if len1 == 0 || len2 == 0 {
		return 0.0
	}

	// Distancia máxima de emparejamiento para considerar que dos caracteres coinciden
	matchDist := int(math.Floor(math.Max(float64(len1), float64(len2))/2.0)) - 1
	if matchDist < 0 {
		matchDist = 0
	}

	s1Matches := make([]bool, len1)
	s2Matches := make([]bool, len2)

	matches := 0
	for i := 0; i < len1; i++ {
		start := int(math.Max(0, float64(i-matchDist)))
		end := int(math.Min(float64(len2-1), float64(i+matchDist)))

		for j := start; j <= end; j++ {
			if s2Matches[j] {
				continue
			}
			if s1[i] == s2[j] {
				s1Matches[i] = true
				s2Matches[j] = true
				matches++
				break
			}
		}
	}

	if matches == 0 {
		return 0.0
	}

	// Calcular transposiciones
	transpositions := 0
	k := 0
	for i := 0; i < len1; i++ {
		if !s1Matches[i] {
			continue
		}
		for !s2Matches[k] {
			k++
		}
		if s1[i] != s2[k] {
			transpositions++
		}
		k++
	}

	m := float64(matches)
	t := float64(transpositions) / 2.0

	// Distancia de Jaro
	jaro := (m/float64(len1) + m/float64(len2) + (m-t)/m) / 3.0

	// Ajuste de Winkler (peso estándar p = 0.1 para prefijos idénticos hasta longitud máxima de 4)
	p := 0.1
	prefixLen := 0
	maxPrefix := int(math.Min(4, math.Min(float64(len1), float64(len2))))
	for i := 0; i < maxPrefix; i++ {
		if s1[i] == s2[i] {
			prefixLen++
		} else {
			break
		}
	}

	return jaro + float64(prefixLen)*p*(1.0-jaro)
}
