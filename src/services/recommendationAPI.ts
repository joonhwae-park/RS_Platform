// Service to call the recommendation API
const RECOMMENDATION_API_URL = import.meta.env.VITE_RECOMMENDATION_API_URL;

export interface RecommendationRequest {
  session_id: string;
  topk_per_model?: number;
  phase?: number;
}

export interface RecommendationResponse {
  session_id: string;
  phase: number;
  svd_top_saved: number;
  p5_top_saved: number;
  svd_top100_size: number;
  display_sequence: Array<[string, string, number]>; // [model, movie_id, display_order]
}

export class RecommendationAPI {
  private static instance: RecommendationAPI;
  
  public static getInstance(): RecommendationAPI {
    if (!RecommendationAPI.instance) {
      RecommendationAPI.instance = new RecommendationAPI();
    }
    return RecommendationAPI.instance;
  }

  async generateRecommendations(sessionId: string): Promise<RecommendationResponse | null> {
    try {
      if (!RECOMMENDATION_API_URL) {
        console.error('VITE_RECOMMENDATION_API_URL is not configured');
        return null;
      }
      
      console.log('Calling recommendation API for session:', sessionId);
      console.log('API URL:', RECOMMENDATION_API_URL);
      
      const requestBody: RecommendationRequest = {
        session_id: sessionId,
        topk_per_model: 10,
        phase: 2
      };

      const response = await fetch(`${RECOMMENDATION_API_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add webhook secret if configured
          ...(import.meta.env.VITE_WEBHOOK_SECRET && {
            'X-Webhook-Secret': import.meta.env.VITE_WEBHOOK_SECRET
          })
        },
        body: JSON.stringify(requestBody),
        // Add timeout and other fetch options
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Recommendation API error:', response.status, errorText);
        return null;
      }

      const result: RecommendationResponse = await response.json();
      console.log('Recommendation API response:', result);
      
      return result;
    } catch (error) {
      console.error('Error calling recommendation API:', error);
      return null;
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      if (!RECOMMENDATION_API_URL) {
        console.error('VITE_RECOMMENDATION_API_URL is not configured');
        return false;
      }
      
      const response = await fetch(`${RECOMMENDATION_API_URL}/health`);
      const isHealthy = response.ok;
      console.log('Health check result:', isHealthy);
      return isHealthy;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

export const recommendationAPI = RecommendationAPI.getInstance();