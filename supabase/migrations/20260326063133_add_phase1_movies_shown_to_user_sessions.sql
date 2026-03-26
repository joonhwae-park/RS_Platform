/*
  # Add phase1_movies_shown to user_sessions

  1. New Columns
    - `phase1_movies_shown` (integer, nullable) - The total number of movies
      displayed to the user during Phase 1 before they moved on.

  2. Purpose
    - Track how many Phase 1 movies each user was exposed to.
    - With the new batch-loading system (up to 100 movies), users may see
      different counts depending on how many "Load More" clicks they made.

  3. Notes
    - Nullable because existing sessions did not record this value.
    - Defaults to NULL; set by the frontend when transitioning out of Phase 1.
*/

ALTER TABLE user_sessions
  ADD COLUMN IF NOT EXISTS phase1_movies_shown integer;

COMMENT ON COLUMN user_sessions.phase1_movies_shown IS 'Total number of Phase 1 movies shown to the user before moving on';
