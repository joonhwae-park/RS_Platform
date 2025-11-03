interface SVDUserFactors {
  [userId: string]: number[];
}

interface SVDItemFactors {
  [movieId: string]: number[];
}

export interface RecommenderConfig {
  // Weight factors for different aspects of recommendation
  ratingWeight: number;
  genreWeight: number;
  yearWeight: number;
  directorWeight: number;
  diversityBoost: number;
}

export class RecommenderService {
  /**
   * Trigger recommendation generation via the Cloud Run API
   * Returns the movie IDs from the API response in display order
   */
  async triggerRecommendationGeneration(sessionId: string): Promise<string[] | null> {
    try {
      console.log('Triggering recommendation generation for session:', sessionId);
      console.log('Environment check - API URL configured:', !!import.meta.env.VITE_RECOMMENDATION_API_URL);

      const result = await recommendationAPI.generateRecommendations(sessionId);

      if (result && result.display_sequence) {
        console.log('Recommendations generated successfully:', result);

        // Extract movie IDs from display_sequence in order
        // display_sequence is Array<[model, movie_id, display_order]>
        // Sort by display_order and extract movie_id
        const movieIds = result.display_sequence
          .sort((a, b) => a[2] - b[2]) // Sort by display_order (index 2)
          .map(item => item[1]); // Extract movie_id (index 1)

        console.log('Extracted movie IDs from API response:', movieIds);
        return movieIds;
      } else {
        console.warn('Failed to generate recommendations via API or no display_sequence returned');
        return null;
      }
    } catch (error) {
      console.error('Error triggering recommendation generation:', error);
      return null;
    }
  }

  /**
   * Get recommendations from the recommendations table in Supabase
   */
  async generateRecommendations(
    sessionId: string,
    userRatings: any[]
  ): Promise<string[]> {
    try {
      console.log('Reading recommendations from database for session:', sessionId);

      // Add initial delay to allow database write to settle after backend API completes
      // This handles database replication lag and connection pooling delays
      console.log('Waiting for database replication to settle...');
      await new Promise(resolve => setTimeout(resolve, 1500)); // 1.5 second initial delay

      // Retry logic to handle backend processing time and database replication lag
      // Try up to 5 times with longer delays to ensure backend completes
      const maxRetries = 5;
      const retryDelay = 1500; // 1500ms between retries (increased from 1000ms)

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        console.log(`Attempt ${attempt}/${maxRetries} to fetch recommendations`);

        // Get recommendations from the recommendations table for THIS session AND phase
        // The backend ensures only one batch exists per session by:
        // 1. Deleting old recommendations before inserting new ones (per session/model/phase)
        // 2. Unique constraint on (session_id, movie_id, model, phase)
        // We fetch only Phase 2 recommendations with display_order for this session
        // and sort by display_order.
        const { data: recommendations, error: fetchError } = await supabase
          .from('recommendations')
          .select('movie_id, display_order')
          .eq('session_id', sessionId)
          .eq('phase', 2)
          .not('display_order', 'is', null)
          .order('display_order', { ascending: true });

        if (fetchError) {
          console.error(`Attempt ${attempt}: Error fetching recommendations:`, fetchError);
          if (attempt === maxRetries) {
            return this.getFallbackRecommendations();
          }
          // Wait before retry
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          continue;
        }

        if (recommendations && recommendations.length > 0) {
          // Extract movie IDs in display_order (should be unique due to unique constraint)
          const movieIds = recommendations.map(r => r.movie_id);

          console.log('✅ Recommendations loaded for session', sessionId, ':', movieIds);
          console.log('Total recommendations with display_order:', recommendations.length);
          console.log('Display orders:', recommendations.map(r => ({ id: r.movie_id, order: r.display_order })));

          return movieIds;
        }

        // No recommendations found, wait and retry unless this is the last attempt
        console.warn(`Attempt ${attempt}: No recommendations found, retrying...`);
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
      }

      // All retries exhausted
      console.warn('All retry attempts exhausted, using fallback');
      return this.getFallbackRecommendations();

    } catch (error) {
      console.error('Error reading recommendations from database:', error);
      return this.getFallbackRecommendations();
    }
  }

  /**
   * Fallback recommendations when SVD algorithm fails
   */
  private async getFallbackRecommendations(): Promise<string[]> {
    try {
      console.log('Using fallback: getting first 10 phase2_movies');
      
      const { data: phase2Movies, error } = await supabase
        .from('phase2_movies')
        .select('id')
        .limit(10);

      if (error) throw error;
      
      const movieIds = phase2Movies?.map(m => m.id) || [];
      
      // Shuffle the array for some randomness
      for (let i = movieIds.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [movieIds[i], movieIds[j]] = [movieIds[j], movieIds[i]];
      }
      
      console.log('Using fallback recommendations:', movieIds);
      return movieIds;
    } catch (error) {
      console.error('Error getting fallback recommendations:', error);
      return [];
    }
  }

  async checkHealth(): Promise<boolean> {
    return await recommendationAPI.checkHealth();
  }
  async logRecommendation(sessionId: string, recommendedMovieIds: string[]) {
    try {
      console.log('Recommendations for session', sessionId, ':', recommendedMovieIds);
      
      // Optional: Store in database for later analysis
      // await supabase.from('recommendations_log').insert({
      //   session_id: sessionId,
      //   recommended_movies: recommendedMovieIds,
      //   algorithm_version: 'SVD-1.0',
      //   created_at: new Date().toISOString()
      // });
    } catch (error) {
      console.error('Error logging recommendation:', error);
    }
  }
}

import { recommendationAPI } from './recommendationAPI';
import { supabase } from '../lib/supabase';
export const recommenderService = new RecommenderService();