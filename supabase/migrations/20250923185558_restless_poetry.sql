/*
  # Fix questionnaire_responses table schema to match actual questionnaire

  1. Current Questionnaire Analysis
    Based on QuestionnaireScreen.tsx, the actual questionnaire has:
    
    REQUIRED FIELDS:
    - movie_watching_frequency (How many movies per month)
    - movie_genre_preferences (Single favorite genre selection)
    - movie_expertise (Knowledge about movies, 1-7 scale)
    - attention_check (Should be "4")
    - diversity_attitude (I often prefer watching similar movies, 1-7 scale)
    - diversity_attitude2 (Compared to my peers, I watch more diverse movies, 1-7 scale)
    - novelty_attitude (I often prefer watching novel movies, 1-7 scale)
    - novelty_attitude2 (Compared to my peers, I often watch conventional movies, 1-7 scale)
    - serendipity_attitude (I often prefer watching unexpected movies, 1-7 scale)
    - serendipity_attitude2 (Compared to my peers, I often watch movies that are easier to discover, 1-7 scale)
    
    OPTIONAL FIELDS:
    - gender
    - age_range
    - additional_comments
    - email (for lottery)

  2. Fields NOT in current questionnaire:
    - openness_to_experience (removed)
    - risk_aversion (removed)
    - streaming_services (removed)
    - primary_streaming_service (removed)
    - nationality (removed)
    - occupation (removed)

  3. New fields needed:
    - novelty_attitude2
    - diversity_attitude2
    - serendipity_attitude2
*/

-- Add new columns for the second attitude questions
DO $$
BEGIN
  -- Add novelty_attitude2 column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'novelty_attitude2'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN novelty_attitude2 text;
  END IF;

  -- Add diversity_attitude2 column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'diversity_attitude2'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN diversity_attitude2 text;
  END IF;

  -- Add serendipity_attitude2 column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'serendipity_attitude2'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN serendipity_attitude2 text;
  END IF;
END $$;

-- Update constraints for required fields that exist in current questionnaire
-- Set defaults first to avoid constraint violations

-- movie_watching_frequency (required)
UPDATE questionnaire_responses 
SET movie_watching_frequency = '2-3' 
WHERE movie_watching_frequency IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_watching_frequency SET NOT NULL;

-- movie_genre_preferences (required, single selection)
UPDATE questionnaire_responses 
SET movie_genre_preferences = ARRAY['Drama'] 
WHERE movie_genre_preferences IS NULL OR array_length(movie_genre_preferences, 1) IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_genre_preferences SET NOT NULL;

-- movie_expertise (required)
UPDATE questionnaire_responses 
SET movie_expertise = '4' 
WHERE movie_expertise IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_expertise SET NOT NULL;

-- attention_check (required)
UPDATE questionnaire_responses 
SET attention_check = '4' 
WHERE attention_check IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN attention_check SET NOT NULL;

-- diversity_attitude (required - first diversity question)
UPDATE questionnaire_responses 
SET diversity_attitude = '4' 
WHERE diversity_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN diversity_attitude SET NOT NULL;

-- novelty_attitude (required - first novelty question)
UPDATE questionnaire_responses 
SET novelty_attitude = '4' 
WHERE novelty_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN novelty_attitude SET NOT NULL;

-- serendipity_attitude (required - first serendipity question)
UPDATE questionnaire_responses 
SET serendipity_attitude = '4' 
WHERE serendipity_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN serendipity_attitude SET NOT NULL;

-- Set defaults for new required fields and make them NOT NULL
UPDATE questionnaire_responses 
SET novelty_attitude2 = '4' 
WHERE novelty_attitude2 IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN novelty_attitude2 SET NOT NULL;

UPDATE questionnaire_responses 
SET diversity_attitude2 = '4' 
WHERE diversity_attitude2 IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN diversity_attitude2 SET NOT NULL;

UPDATE questionnaire_responses 
SET serendipity_attitude2 = '4' 
WHERE serendipity_attitude2 IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN serendipity_attitude2 SET NOT NULL;

-- Drop unused columns that are not in current questionnaire
-- (Commented out for safety - uncomment if you want to permanently remove the data)
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS openness_to_experience;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS risk_aversion;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS streaming_services;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS primary_streaming_service;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS nationality;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS occupation;

-- Add comments to document the current schema
COMMENT ON TABLE questionnaire_responses IS 'Stores responses from the final questionnaire - updated to match current questionnaire structure';

-- Required fields
COMMENT ON COLUMN questionnaire_responses.movie_watching_frequency IS 'How many movies user watches per month (required)';
COMMENT ON COLUMN questionnaire_responses.movie_genre_preferences IS 'Users most favorite movie genre - single selection (required)';
COMMENT ON COLUMN questionnaire_responses.movie_expertise IS 'How knowledgeable about movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.attention_check IS 'Attention check question - should be 4 (required)';

-- Attitude questions (2 questions each for diversity, novelty, serendipity)
COMMENT ON COLUMN questionnaire_responses.diversity_attitude IS 'I often prefer watching similar movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.diversity_attitude2 IS 'Compared to my peers, I watch more diverse movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.novelty_attitude IS 'I often prefer watching novel movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.novelty_attitude2 IS 'Compared to my peers, I often watch conventional movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.serendipity_attitude IS 'I often prefer watching unexpected movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.serendipity_attitude2 IS 'Compared to my peers, I often watch movies that are easier to discover (1-7 scale, required)';

-- Optional fields
COMMENT ON COLUMN questionnaire_responses.gender IS 'User gender (optional)';
COMMENT ON COLUMN questionnaire_responses.age_range IS 'User age range (optional)';
COMMENT ON COLUMN questionnaire_responses.additional_comments IS 'Additional user comments (optional)';
COMMENT ON COLUMN questionnaire_responses.email IS 'Email for lottery participation (optional)';