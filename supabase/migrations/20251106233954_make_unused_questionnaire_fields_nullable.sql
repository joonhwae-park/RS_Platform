/*
  # Make unused questionnaire fields nullable

  1. Issue
    - The questionnaire form does not collect openness_to_experience or risk_aversion
    - These columns are marked as NOT NULL in the database
    - This causes INSERT operations to fail when submitting the questionnaire
    
  2. Changes
    - Make openness_to_experience nullable
    - Make risk_aversion nullable
    
  3. Rationale
    - These fields were removed from the questionnaire per migration 20250923185558
    - They should have been made nullable or dropped at that time
    - Making them nullable allows questionnaire submissions to succeed
*/

-- Make openness_to_experience nullable
ALTER TABLE questionnaire_responses 
ALTER COLUMN openness_to_experience DROP NOT NULL;

-- Make risk_aversion nullable
ALTER TABLE questionnaire_responses 
ALTER COLUMN risk_aversion DROP NOT NULL;

-- Update comments to reflect that these fields are legacy/unused
COMMENT ON COLUMN questionnaire_responses.openness_to_experience IS 'LEGACY FIELD - Not used in current questionnaire (nullable)';
COMMENT ON COLUMN questionnaire_responses.risk_aversion IS 'LEGACY FIELD - Not used in current questionnaire (nullable)';
