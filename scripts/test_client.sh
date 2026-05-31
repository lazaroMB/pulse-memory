#!/usr/bin/env bash

# Exit immediately if any command fails
# We avoid setting 'set -e' so the interactive loop continues even if a network request fails

SERVER_URL="http://localhost:8080"
SESSION_ID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null | cut -c 1-8 || echo "session-$(date +%s)")
ENTITY_ID="c33f20d5-5d9c-4972-bb2f-34d380963579"
AGENT_ROLE="developer_agent"

# Parse CLI arguments to control fact retrieval
INCLUDE_FACTS="false"
for arg in "$@"; do
    if [[ "$arg" == "--include-facts" || "$arg" == "--facts" || "$arg" == "-f" ]]; then
        INCLUDE_FACTS="true"
    fi
done

# ANSI Terminal Colors
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
GRAY="\033[0;90m"
RED="\033[0;31m"
NC="\033[0m" # No Color
BOLD="\033[1m"

echo -e "${BOLD}=========================================================${NC}"
echo -e "${BOLD}       Multi-Agent Swarm Memory Interactive Chat         ${NC}"
echo -e "${BOLD}=========================================================${NC}"
echo -e "${GRAY}Server URL:    ${SERVER_URL}${NC}"
echo -e "${GRAY}Session ID:    ${SESSION_ID}${NC}"
echo -e "${GRAY}Entity ID:     ${ENTITY_ID} (John)${NC}"
echo -e "${GRAY}Agent Role:    ${AGENT_ROLE}${NC}"
echo -e "${GRAY}Include Facts: ${YELLOW}${INCLUDE_FACTS}${NC} (pass --facts, --include-facts or -f to enable)"
echo -e "${GRAY}Commands:      /exit, /quit (to exit)${NC}"
echo -e "${GRAY}               /relation <type> <target_id> (to link entity)${NC}"
echo -e "${BOLD}=========================================================${NC}"

# Check server health
if ! curl -s -f "$SERVER_URL/health" >/dev/null; then
    echo -e "${RED}Error: Server is not running on ${SERVER_URL}.${NC}"
    echo -e "${RED}Please start the Go server first by running 'make run'.${NC}"
    exit 1
fi

while true; do
    echo -e -n "\n${GREEN}${BOLD}You > ${NC}"
    read -r USER_INPUT

    # Handle exit conditions
    if [[ "$USER_INPUT" == "/exit" || "$USER_INPUT" == "/quit" || "$USER_INPUT" == "exit" || "$USER_INPUT" == "quit" ]]; then
        echo -e "${GRAY}Exiting chat session. Goodbye!${NC}"
        break
    fi

    # Skip empty input
    if [[ -z "$USER_INPUT" ]]; then
        continue
    fi

    # Handle relationship shortcut: /relation <type> <target_id>
    if [[ "$USER_INPUT" =~ ^/relation[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)$ ]]; then
        REL_TYPE="${BASH_REMATCH[1]}"
        TARGET_ID="${BASH_REMATCH[2]}"
        
        echo -e "${GRAY}Registering graph relationship edge: Entity -[${REL_TYPE}]-> ${TARGET_ID}...${NC}"
        
        REL_RESP=$(curl -s -X POST "$SERVER_URL/relation" \
          -H "Content-Type: application/json" \
          -d "{
            \"source_id\": \"$ENTITY_ID\",
            \"target_id\": \"$TARGET_ID\",
            \"type\": \"$REL_TYPE\"
          }")
        
        STATUS=$(echo "$REL_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'error'))" 2>/dev/null)
        if [[ "$STATUS" == "success" ]]; then
            echo -e "${YELLOW}Relationship registered successfully in temporal graph!${NC}"
        else
            echo -e "${RED}Failed to register relationship: ${REL_RESP}${NC}"
        fi
        continue
    fi

    # Standard Chat Message Request
    RESPONSE=$(curl -s -X POST "$SERVER_URL/chat" \
      -H "Content-Type: application/json" \
      -d "{
        \"session_id\": \"$SESSION_ID\",
        \"entity_id\": \"$ENTITY_ID\",
        \"agent_role\": \"$AGENT_ROLE\",
        \"message\": \"$USER_INPUT\",
        \"includeFacts\": $INCLUDE_FACTS
      }")

    if [[ -z "$RESPONSE" ]]; then
        echo -e "${RED}Error: Server returned empty response.${NC}"
        continue
    fi

    # Parse JSON values using Python for maximum portability
    REPLY=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('responseMessage', ''))" 2>/dev/null)
    FACTS=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    entity_facts = data.get('entityFacts', []) or []
    doc_facts = data.get('documentFacts', []) or []
    facts = entity_facts + doc_facts
    if not facts:
        print('None')
    else:
        print('\n'.join([f'  • [{f.get(\"source_agent\", \"\")}] {f.get(\"attribute\", \"\")}: {f.get(\"value\", \"\")}' for f in facts]))
except Exception:
    print('None')
" 2>/dev/null)

    # Print the Agent response
    echo -e "\n${BLUE}${BOLD}Agent > ${NC}${REPLY}"

    # Print any long-term memories retrieved
    if [[ "$FACTS" != "None" && ! -z "$FACTS" ]]; then
        echo -e "\n${YELLOW}${BOLD}[Retrieved Long-Term Memories]${NC}"
        echo -e "${YELLOW}${FACTS}${NC}"
    else
        echo -e "\n${GRAY}[No Long-Term Memories Retrieved]${NC}"
    fi
    
    echo -e "${GRAY}(Full response: $RESPONSE)${NC}"
    echo -e "${GRAY}---------------------------------------------------------${NC}"
done
