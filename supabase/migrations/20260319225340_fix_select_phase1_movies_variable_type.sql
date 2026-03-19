/*
  # Fix select_phase1_movies_for_session function

  ## Problem
  The v_existing_count variable was declared as `integer` but the batch-existence
  check assigned an `array_agg(movie_id)` result into it, then called
  `array_length(v_existing_count, 1)` — PostgreSQL resolves this as
  `array_length(integer, integer)` which does not exist, causing the error.

  ## Fix
  Use a dedicated `text[]` variable (v_existing_batch) to hold the result of the
  array_agg check, and keep v_existing_count as an integer only used for the
  COUNT(*) check (which was the original intent).
*/

CREATE OR REPLACE FUNCTION select_phase1_movies_for_session(p_session_id uuid, p_batch_number integer DEFAULT 1)
RETURNS text[]
LANGUAGE plpgsql
AS $$
DECLARE
  v_existing_batch text[];
  v_vote_count_movies text[];
  v_vote_variance_movies text[];
  v_genre_movies text[];
  v_all_movies text[];
  v_movie_id text;
  v_genre_num integer;
  v_excluded_movies text[];
  v_limit_per_step integer;
BEGIN
  -- Validate batch number (1-4 for up to 100 movies)
  IF p_batch_number < 1 OR p_batch_number > 4 THEN
    RETURN ARRAY[]::text[];
  END IF;

  -- Check if batch already exists for this session
  SELECT array_agg(movie_id ORDER BY created_at)
  INTO v_existing_batch
  FROM session_phase1_movies
  WHERE session_id = p_session_id
    AND selection_type LIKE 'batch_' || p_batch_number || '_%';

  IF v_existing_batch IS NOT NULL AND array_length(v_existing_batch, 1) > 0 THEN
    RETURN v_existing_batch;
  END IF;

  -- Get all previously selected movies to exclude from new batch
  SELECT COALESCE(array_agg(movie_id), ARRAY[]::text[])
  INTO v_excluded_movies
  FROM session_phase1_movies
  WHERE session_id = p_session_id;

  -- Batch 4 loads only 10 movies (to reach 100 max), others load 10+10+10
  v_limit_per_step := CASE WHEN p_batch_number = 4 THEN 3 ELSE 10 END;

  -- Step 1: Select movies weighted by vote_count from top_vote_count = 1
  SELECT array_agg(id)
  INTO v_vote_count_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_count = 1
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT v_limit_per_step
  ) t;

  -- Insert vote_count selections
  IF v_vote_count_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_count_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_vote_count');
    END LOOP;
  END IF;

  -- Update excluded movies for next step
  v_excluded_movies := v_excluded_movies || COALESCE(v_vote_count_movies, ARRAY[]::text[]);

  -- Step 2: Select movies weighted by vote_count from top_vote_variance = 1
  SELECT array_agg(id)
  INTO v_vote_variance_movies
  FROM (
    SELECT id
    FROM movies
    WHERE top_vote_variance = 1
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT v_limit_per_step
  ) t;

  -- Insert vote_variance selections
  IF v_vote_variance_movies IS NOT NULL THEN
    FOREACH v_movie_id IN ARRAY v_vote_variance_movies
    LOOP
      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_vote_variance');
    END LOOP;
  END IF;

  -- Update excluded movies for next step
  v_excluded_movies := v_excluded_movies || COALESCE(v_vote_variance_movies, ARRAY[]::text[]);

  -- Step 3: Select movies weighted by vote_count per genre
  v_genre_movies := ARRAY[]::text[];
  FOR v_genre_num IN 1..10 LOOP
    SELECT id INTO v_movie_id
    FROM movies
    WHERE top_genre = v_genre_num
      AND id != ALL(v_excluded_movies)
    ORDER BY random() * NULLIF(vote_count, 0) DESC, random()
    LIMIT 1;

    IF v_movie_id IS NOT NULL THEN
      v_genre_movies := array_append(v_genre_movies, v_movie_id);

      INSERT INTO session_phase1_movies (session_id, movie_id, selection_type)
      VALUES (p_session_id, v_movie_id, 'batch_' || p_batch_number || '_genre_' || v_genre_num);

      v_excluded_movies := v_excluded_movies || ARRAY[v_movie_id];
      v_movie_id := NULL;
    END IF;
  END LOOP;

  -- Combine all movies for this batch
  v_all_movies := COALESCE(v_vote_count_movies, ARRAY[]::text[]) ||
                  COALESCE(v_vote_variance_movies, ARRAY[]::text[]) ||
                  v_genre_movies;

  RETURN v_all_movies;
END;
$$;
