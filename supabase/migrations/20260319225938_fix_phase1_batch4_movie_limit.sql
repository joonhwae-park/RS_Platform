/*
  # Fix Phase 1 Batch 4 Movie Limit

  ## Problem
  Batch 4 used v_limit_per_step = 3, which means it selects up to 3+3+10 = ~16 movies,
  not the intended 10 to reach a 100-movie total.

  ## Fix
  Rewrite batch 4 logic to select exactly up to `v_remaining` movies total across all
  steps, capping the overall total at 100. The approach:
  - Track total movies already loaded for the session.
  - Compute how many remain to reach 100.
  - Cap each step proportionally, and limit the entire batch to `v_remaining`.
  - Batches 1–3 still load ~30 movies each.

  ## Changes
  - Updated `select_phase1_movies_for_session` function to enforce a 100-movie cap.
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
  v_total_loaded integer;
  v_remaining integer;
  v_limit_per_step integer;
  v_genre_limit integer;
  v_batch_limit integer;
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

  -- Count how many movies have already been loaded
  SELECT COUNT(*) INTO v_total_loaded
  FROM session_phase1_movies
  WHERE session_id = p_session_id;

  -- Compute how many more we can load (cap total at 100)
  v_remaining := 100 - v_total_loaded;

  IF v_remaining <= 0 THEN
    RETURN ARRAY[]::text[];
  END IF;

  -- For batches 1-3, each batch targets 30 movies (10 vote_count + 10 vote_variance + 10 genre)
  -- For batch 4, only load whatever is remaining (capped at 10)
  v_batch_limit := LEAST(30, v_remaining);

  -- Distribute the batch_limit: split into thirds, genre gets the rest
  v_limit_per_step := v_batch_limit / 3;
  v_genre_limit := v_batch_limit - (v_limit_per_step * 2);

  -- Step 1: Select movies from top_vote_count = 1
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

  -- Step 2: Select movies from top_vote_variance = 1
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

  -- Step 3: Select movies per genre, up to v_genre_limit total
  v_genre_movies := ARRAY[]::text[];
  FOR v_genre_num IN 1..10 LOOP
    EXIT WHEN array_length(COALESCE(v_genre_movies, ARRAY[]::text[]), 1) >= v_genre_limit;

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
