/*
  # Update questionnaire_responses table schema to match current questionnaire

  1. Schema Changes
    - Remove unused columns that are no longer in the questionnaire
    - Ensure all current questionnaire fields are properly represented
    - Update column constraints and types as needed

  2. Current Questionnaire Fields (from QuestionnaireScreen.tsx):
    - movieWatchingFrequency (required)
    - movieGenrePreferences (required, single selection now)
    - opennessToExperience (required, 1-7 scale)
    - riskAversion (required, 1-7 scale) 
    - movieExpertise (required, 1-7 scale)
    - attentionCheck (required, should be "4")
    - serendipityAttitude (required, 1-7 scale)
    - noveltyAttitude (required, 1-7 scale)
    - diversityAttitude (required, 1-7 scale)
    - gender (optional)
    - ageRange (optional)
    - nationality (optional) - removed from questionnaire
    - occupation (optional) - removed from questionnaire
    - additionalComments (optional)
    - email (optional)

  3. Removed Fields
    - streaming_services (no longer in questionnaire)
    - primary_streaming_service (no longer in questionnaire)
    - nationality (removed from questionnaire)
    - occupation (removed from questionnaire)

  4. Updated Constraints
    - Make required fields NOT NULL with appropriate defaults
    - Ensure text fields for Likert scales (1-7)
*/

-- First, let's add any missing columns that might be needed
DO $$
BEGIN
  -- Ensure all current questionnaire fields exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'movie_watching_frequency'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN movie_watching_frequency text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'movie_genre_preferences'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN movie_genre_preferences text[];
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'openness_to_experience'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN openness_to_experience text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'risk_aversion'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN risk_aversion text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'movie_expertise'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN movie_expertise text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'attention_check'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN attention_check text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'serendipity_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN serendipity_attitude text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'novelty_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN novelty_attitude text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'diversity_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN diversity_attitude text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'gender'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN gender text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'age_range'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN age_range text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'additional_comments'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN additional_comments text;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'email'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN email text;
  END IF;
END $$;

-- Update constraints for required fields (make them NOT NULL with defaults for existing records)
-- Note: We'll set defaults first, then make them NOT NULL

-- Update movie_watching_frequency to be required
UPDATE questionnaire_responses 
SET movie_watching_frequency = '2-3' 
WHERE movie_watching_frequency IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_watching_frequency SET NOT NULL;

-- Update movie_genre_preferences to be required (single selection now)
UPDATE questionnaire_responses 
SET movie_genre_preferences = ARRAY['Drama'] 
WHERE movie_genre_preferences IS NULL OR array_length(movie_genre_preferences, 1) IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_genre_preferences SET NOT NULL;

-- Update psychological measures to be required
UPDATE questionnaire_responses 
SET openness_to_experience = '4' 
WHERE openness_to_experience IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN openness_to_experience SET NOT NULL;

UPDATE questionnaire_responses 
SET risk_aversion = '4' 
WHERE risk_aversion IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN risk_aversion SET NOT NULL;

UPDATE questionnaire_responses 
SET movie_expertise = '4' 
WHERE movie_expertise IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN movie_expertise SET NOT NULL;

-- Update attention check to be required
UPDATE questionnaire_responses 
SET attention_check = '4' 
WHERE attention_check IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN attention_check SET NOT NULL;

-- Update attitude measures to be required
UPDATE questionnaire_responses 
SET serendipity_attitude = '4' 
WHERE serendipity_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN serendipity_attitude SET NOT NULL;

UPDATE questionnaire_responses 
SET novelty_attitude = '4' 
WHERE novelty_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN novelty_attitude SET NOT NULL;

UPDATE questionnaire_responses 
SET diversity_attitude = '4' 
WHERE diversity_attitude IS NULL;

ALTER TABLE questionnaire_responses 
ALTER COLUMN diversity_attitude SET NOT NULL;

-- Drop columns that are no longer used in the current questionnaire
-- Note: Be careful with this in production - you might want to keep the data

-- Commenting out the drops for safety - uncomment if you're sure you want to remove the data
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS streaming_services;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS primary_streaming_service;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS nationality;
-- ALTER TABLE questionnaire_responses DROP COLUMN IF EXISTS occupation;

-- Add comments to document the current schema
COMMENT ON TABLE questionnaire_responses IS 'Stores responses from the final questionnaire completed by users';
COMMENT ON COLUMN questionnaire_responses.movie_watching_frequency IS 'How many movies user watches per month (required)';
COMMENT ON COLUMN questionnaire_responses.movie_genre_preferences IS 'Users most favorite movie genre - single selection (required)';
COMMENT ON COLUMN questionnaire_responses.openness_to_experience IS 'I often prefer watching similar movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.risk_aversion IS 'Compared to my peers, I watch more diverse movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.movie_expertise IS 'How knowledgeable about movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.attention_check IS 'Attention check question - should be 4 (required)';
COMMENT ON COLUMN questionnaire_responses.serendipity_attitude IS 'I often prefer watching novel movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.novelty_attitude IS 'Compared to my peers, I often watch conventional movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.diversity_attitude IS 'I often prefer watching unexpected movies (1-7 scale, required)';
COMMENT ON COLUMN questionnaire_responses.gender IS 'User gender (optional)';
COMMENT ON COLUMN questionnaire_responses.age_range IS 'User age range (optional)';
COMMENT ON COLUMN questionnaire_responses.additional_comments IS 'Additional user comments (optional)';
COMMENT ON COLUMN questionnaire_responses.email IS 'Email for lottery participation (optional)';