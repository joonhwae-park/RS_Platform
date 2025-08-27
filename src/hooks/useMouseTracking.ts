import { useEffect, useRef } from 'react';
import { supabase } from '../lib/supabase';

interface MouseEvent {
  x: number;
  y: number;
  eventType: 'move' | 'click' | 'scroll';
  timestamp: string;
  pageUrl: string;
  elementTarget?: string;
}

export const useMouseTracking = (sessionId: string | null, isActive: boolean = true) => {
  const lastSaveTime = useRef<number>(0);
  const mouseEvents = useRef<MouseEvent[]>([]);
  const saveInterval = useRef<NodeJS.Timeout | null>(null);

  const saveEventsToDatabase = async (events: MouseEvent[]) => {
    if (!sessionId || sessionId.startsWith('local_') || events.length === 0) {
      return;
    }

    try {
      const eventsToSave = events.map(event => ({
        session_id: sessionId,
        x: event.x,
        y: event.y,
        event_type: event.eventType,
        timestamp: event.timestamp,
        page_url: event.pageUrl,
        element_target: event.elementTarget || ''
      }));

      const { error } = await supabase
        .from('mouse_events')
        .insert(eventsToSave);

      if (error) {
        console.error('Error saving mouse events:', error);
      } else {
        console.log(`Saved ${events.length} mouse events to database`);
      }
    } catch (error) {
      console.error('Error in saveEventsToDatabase:', error);
    }
  };

  const addMouseEvent = (event: MouseEvent) => {
    mouseEvents.current.push(event);
    
    // Save events in batches of 50 or every 5 seconds
    if (mouseEvents.current.length >= 50) {
      saveEventsToDatabase([...mouseEvents.current]);
      mouseEvents.current = [];
    }
  };

  const getElementDescription = (element: Element): string => {
    const tagName = element.tagName.toLowerCase();
    
    // Handle both HTML and SVG elements - SVG className is SVGAnimatedString
    let className = '';
    if (element.className) {
      const classValue = typeof element.className === 'string' 
        ? element.className 
        : element.className.baseVal || '';
      
      if (classValue) {
        const classes = classValue.split(' ').filter(cls => cls.trim() !== '');
        className = classes.length > 0 ? `.${classes.join('.')}` : '';
      }
    }
    
    const id = element.id ? `#${element.id}` : '';
    return `${tagName}${id}${className}`.substring(0, 100);
  };

  useEffect(() => {
    if (!isActive || !sessionId) return;

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const now = Date.now();
      // Throttle mouse move events to every 100ms
      if (now - lastSaveTime.current < 100) return;
      
      lastSaveTime.current = now;
      
      addMouseEvent({
        x: e.clientX,
        y: e.clientY,
        eventType: 'move',
        timestamp: new Date().toISOString(),
        pageUrl: window.location.pathname,
        elementTarget: e.target ? getElementDescription(e.target as Element) : undefined
      });
    };

    const handleMouseClick = (e: globalThis.MouseEvent) => {
      addMouseEvent({
        x: e.clientX,
        y: e.clientY,
        eventType: 'click',
        timestamp: new Date().toISOString(),
        pageUrl: window.location.pathname,
        elementTarget: e.target ? getElementDescription(e.target as Element) : undefined
      });
    };

    const handleScroll = () => {
      addMouseEvent({
        x: window.scrollX,
        y: window.scrollY,
        eventType: 'scroll',
        timestamp: new Date().toISOString(),
        pageUrl: window.location.pathname
      });
    };

    // Add event listeners
    document.addEventListener('mousemove', handleMouseMove, { passive: true });
    document.addEventListener('click', handleMouseClick, { passive: true });
    document.addEventListener('scroll', handleScroll, { passive: true });

    // Set up periodic save interval (every 5 seconds)
    saveInterval.current = setInterval(() => {
      if (mouseEvents.current.length > 0) {
        saveEventsToDatabase([...mouseEvents.current]);
        mouseEvents.current = [];
      }
    }, 5000);

    // Cleanup function
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('click', handleMouseClick);
      document.removeEventListener('scroll', handleScroll);
      
      if (saveInterval.current) {
        clearInterval(saveInterval.current);
      }
      
      // Save any remaining events before cleanup
      if (mouseEvents.current.length > 0) {
        saveEventsToDatabase([...mouseEvents.current]);
      }
    };
  }, [sessionId, isActive]);

  // Save events when component unmounts or session changes
  useEffect(() => {
    return () => {
      if (mouseEvents.current.length > 0) {
        saveEventsToDatabase([...mouseEvents.current]);
      }
    };
  }, [sessionId]);

  return {
    saveRemainingEvents: () => {
      if (mouseEvents.current.length > 0) {
        saveEventsToDatabase([...mouseEvents.current]);
        mouseEvents.current = [];
      }
    }
  };
};