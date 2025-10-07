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
        console.error('VITE_RECOMMENDATION_API_URL is not configured. Please set this environment variable.');
        console.log('Current RECOMMENDATION_API_URL:', RECOMMENDATION_API_URL);
        return null;
      }
      
      console.log('Calling recommendation API for session:', sessionId);
      console.log('API URL:', RECOMMENDATION_API_URL);
      console.log('Full API endpoint:', `${RECOMMENDATION_API_URL}/recommend`);
      
      const requestBody: RecommendationRequest = {
        session_id: sessionId,
        topk_per_model: 10,
        phase: 2
      };

      console.log('Request body:', JSON.stringify(requestBody, null, 2));

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

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Recommendation API error:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText,
          url: `${RECOMMENDATION_API_URL}/recommend`
        });
        return null;
      }

      const result: RecommendationResponse = await response.json();
      console.log('Recommendation API response:', result);
      
      return result;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('Recommendation API request timed out after 30 seconds');
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        console.error('Network error - cannot reach recommendation API:', error.message);
        console.error('Check if the API URL is correct and the service is running');
      } else {
        console.error('Error calling recommendation API:', error);
      }
      return null;
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      if (!RECOMMENDATION_API_URL) {
        console.error('VITE_RECOMMENDATION_API_URL is not configured for health check');
        return false;
      }
      
      console.log('Checking API health at:', `${RECOMMENDATION_API_URL}/health`);
      
      const response = await fetch(`${RECOMMENDATION_API_URL}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(10000) // 10 second timeout for health check
      });
      
      const isHealthy = response.ok;
      
      if (isHealthy) {
        const healthData = await response.json();
        console.log('Health check successful:', healthData);
      } else {
        const errorText = await response.text();
        console.error('Health check failed:', {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText
        });
      }
      
      return isHealthy;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('Health check timed out after 10 seconds');
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        console.error('Network error during health check - cannot reach API:', error.message);
      } else {
        console.error('Health check failed:', error);
      }
      return false;
    }
  }
}

export const recommendationAPI = RecommendationAPI.getInstance();