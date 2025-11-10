/*
  # Add phase-specific timestamps to user_sessions

  1. New Columns
    - intro_start_time (timestamptz) - When user started the intro phase
    - initial_start_time (timestamptz) - When user started Phase 1 (initial rating)
    - choice_start_time (timestamptz) - When user reached the choice screen
    - recommendation_start_time (timestamptz) - When user started Phase 2 (recommendations)
    - questionnaire_start_time (timestamptz) - When user started the questionnaire
    
  2. Purpose
    - Track when each phase of the study begins for each user
    - Enable analysis of time spent in each phase
    - Support research on user behavior patterns
    
  3. Notes
    - All timestamps are nullable (phases may not be reached by all users)
    - Existing start_time column represents overall session start
*/

-- Add phase-specific timestamp columns
ALTER TABLE user_sessions 
  ADD COLUMN IF NOT EXISTS intro_start_time timestamptz,
  ADD COLUMN IF NOT EXISTS initial_start_time timestamptz,
  ADD COLUMN IF NOT EXISTS choice_start_time timestamptz,
  ADD COLUMN IF NOT EXISTS recommendation_start_time timestamptz,
  ADD COLUMN IF NOT EXISTS questionnaire_start_time timestamptz;

-- Add comments to document the columns
COMMENT ON COLUMN user_sessions.intro_start_time IS 'Timestamp when user started the intro phase';
COMMENT ON COLUMN user_sessions.initial_start_time IS 'Timestamp when user started Phase 1 (initial movie ratings)';
COMMENT ON COLUMN user_sessions.choice_start_time IS 'Timestamp when user reached the choice screen between Phase 1 and Phase 2';
COMMENT ON COLUMN user_sessions.recommendation_start_time IS 'Timestamp when user started Phase 2 (rating recommendations)';
COMMENT ON COLUMN user_sessions.questionnaire_start_time IS 'Timestamp when user started the final questionnaire';

-- Update existing column comment for clarity
COMMENT ON COLUMN user_sessions.start_time IS 'Overall session start time (created_at alias, kept for backward compatibility)';
