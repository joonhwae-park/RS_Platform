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
   */
  async triggerRecommendationGeneration(sessionId: string): Promise<boolean> {
    try {
      console.log('Triggering recommendation generation for session:', sessionId);
      console.log('Environment check - API URL configured:', !!import.meta.env.VITE_RECOMMENDATION_API_URL);
      
      const result = await recommendationAPI.generateRecommendations(sessionId);
      
      if (result) {
        console.log('Recommendations generated successfully:', result);
        return true;
      } else {
        console.warn('Failed to generate recommendations via API');
        return false;
      }
    } catch (error) {
      console.error('Error triggering recommendation generation:', error);
      return false;
    }
  }

  /**
   * Get recommendations from the recommendations table in Supabase
   */
  async generateRecommendations(
    sessionId: string,
    userRatings: any[]
  ): Promise<number[]> {
    try {
      console.log('Reading recommendations from database for session:', sessionId);
      
      // Get recommendations from the recommendations table for THIS session only
      // The backend ensures only one batch exists per session by:
      // 1. Deleting old recommendations before inserting new ones (per session/model/phase)
      // 2. Unique constraint on (session_id, movie_id, model, phase)
      // Therefore, we can simply fetch all recommendations with display_order for this session
      // and sort by display_order.
      const { data: recommendations, error: fetchError } = await supabase
        .from('recommendations')
        .select('movie_id, display_order')
        .eq('session_id', sessionId)
        .not('display_order', 'is', null)
        .order('display_order', { ascending: true });

      if (fetchError) {
        console.error('Error fetching recommendations from database:', fetchError);
        return this.getFallbackRecommendations();
      }

      if (!recommendations || recommendations.length === 0) {
        console.warn('No recommendations found in database for session:', sessionId);
        return this.getFallbackRecommendations();
      }

      // Extract movie IDs in display_order (should be unique due to unique constraint)
      const movieIds = recommendations.map(r => r.movie_id);

      console.log('✅ Recommendations loaded for session', sessionId, ':', movieIds);
      console.log('Total recommendations with display_order:', recommendations.length);
      console.log('Display orders:', recommendations.map(r => ({ id: r.movie_id, order: r.display_order })));

      return movieIds;

    } catch (error) {
      console.error('Error reading recommendations from database:', error);
      return this.getFallbackRecommendations();
    }
  }

  /**
   * Fallback recommendations when SVD algorithm fails
   */
  private async getFallbackRecommendations(): Promise<number[]> {
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
  async logRecommendation(sessionId: string, recommendedMovieIds: number[]) {
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