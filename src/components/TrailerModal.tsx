import React from 'react';
import { X } from 'lucide-react';

interface TrailerModalProps {
  isOpen: boolean;
  onClose: () => void;
  trailerUrl: string;
  movieTitle: string;
}

export const TrailerModal: React.FC<TrailerModalProps> = ({
  isOpen,
  onClose,
  trailerUrl,
  movieTitle
}) => {
  if (!isOpen) return null;

  // Convert various URL formats to embed format
  const getEmbedUrl = (url: string) => {
    // Vimeo embed HTML (check for iframe with vimeo player)
    if (url.includes('player.vimeo.com/video/')) {
      // Extract video ID from Vimeo player URL
      const match = url.match(/player\.vimeo\.com\/video\/(\d+)/);
      if (match) {
        const videoId = match[1];
        return `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479&texttrack=en`;
      }
    }
    
    // Vimeo direct links
    if (url.includes('vimeo.com/') && !url.includes('player.vimeo.com')) {
      const videoId = url.split('vimeo.com/')[1]?.split('?')[0];
      if (videoId) {
        return `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479&texttrack=en`;
      }
    }
    
    // Google Drive links
    if (url.includes('drive.google.com/file/d/')) {
      const fileId = url.match(/\/file\/d\/([a-zA-Z0-9_-]+)/)?.[1];
      if (fileId) {
        return `https://drive.google.com/file/d/${fileId}/preview?usp=sharing&cc_load_policy=1&hl=en&cc_lang_pref=en&autoplay=0`;
      }
    }
    
    // Google Drive direct preview links
    if (url.includes('drive.google.com') && url.includes('/preview')) {
      let baseUrl = url.includes('usp=sharing') ? url : `${url}${url.includes('?') ? '&' : '?'}usp=sharing`;
      
      // Add subtitle parameters if not present
      if (!baseUrl.includes('cc_load_policy=1')) {
        baseUrl += '&cc_load_policy=1';
      }
      if (!baseUrl.includes('hl=en')) {
        baseUrl += '&hl=en';
      }
      if (!baseUrl.includes('cc_lang_pref=en')) {
        baseUrl += '&cc_lang_pref=en';
      }
      if (!baseUrl.includes('autoplay=0')) {
        baseUrl += '&autoplay=0';
      }
      
      return baseUrl;
    }
    
    // Google Drive view links
    if (url.includes('drive.google.com/open?id=')) {
      const fileId = url.match(/id=([a-zA-Z0-9_-]+)/)?.[1];
      if (fileId) {
        return `https://drive.google.com/file/d/${fileId}/preview?usp=sharing&cc_load_policy=1&hl=en&cc_lang_pref=en&autoplay=0`;
      }
    }
    
    // YouTube embed links (already in embed format)
    if (url.includes('youtube.com/embed/')) {
      return url.includes('cc_load_policy=1') ? url : `${url}${url.includes('?') ? '&' : '?'}cc_load_policy=1&cc_lang_pref=en`;
    }
    
    // YouTube watch links
    if (url.includes('youtube.com/watch?v=')) {
      const videoId = url.split('v=')[1]?.split('&')[0];
      return `https://www.youtube.com/embed/${videoId}?cc_load_policy=1&cc_lang_pref=en`;
    }
    
    // YouTube short links
    if (url.includes('youtu.be/')) {
      const videoId = url.split('youtu.be/')[1]?.split('?')[0];
      return `https://www.youtube.com/embed/${videoId}?cc_load_policy=1&cc_lang_pref=en`;
    }
    
    // Return original URL if no conversion needed
    return url;
  };

  const isVimeoUrl = (url: string) => {
    return url.includes('vimeo.com') || url.includes('player.vimeo.com');
  };
  
  const isGoogleDriveUrl = (url: string) => {
    return url.includes('drive.google.com');
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <h3 className="text-xl font-semibold text-white">{movieTitle} - Trailer</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-800 rounded-lg"
          >
            <X size={24} />
          </button>
        </div>
        <div className="aspect-video relative">
          <iframe
            src={getEmbedUrl(trailerUrl)}
            title={`${movieTitle} Trailer`}
            className="absolute top-0 left-0 w-full h-full"
            frameBorder="0"
            allow={
              isVimeoUrl(trailerUrl) 
                ? "autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share"
                : isGoogleDriveUrl(trailerUrl)
                ? "autoplay; fullscreen; picture-in-picture"
                : "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            }
            allowFullScreen
            sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
          />
        </div>
      </div>
      {isVimeoUrl(trailerUrl) && (
        <script src="https://player.vimeo.com/api/player.js" async />
      )}
    </div>
  );
};