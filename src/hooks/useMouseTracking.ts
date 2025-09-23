import { useEffect, useRef } from 'react';
import { supabase } from '../lib/supabase';

interface MouseEvent {
  x: number;
  y: number;
  timestamp: string;
}

export const useMouseTracking = (sessionId: string | null, isActive: boolean = true) => {
  const eventsBuffer = useRef<MouseEvent[]>([]);
  const lastSaveTime = useRef<number>(Date.now());
  const saveInterval = useRef<NodeJS.Timeout | null>(null);

  const saveEventsToDatabase = async (events: MouseEvent[]) => {
    if (!sessionId || sessionId.startsWith('local_') || events.length === 0) {
      return;
    }

    try {
      const eventsToSave = events.map(event => ({
        session_id: sessionId,
        x: Math.round(event.x), // Ensure integer values
        y: Math.round(event.y), // Ensure integer values
        timestamp: event.timestamp
      }));

      const { error } = await supabase
        .from('mouse_events')
        .insert(eventsToSave);

      if (error) {
        console.error('Error saving mouse events:', error);
      } else {
        console.log(`Saved ${eventsToSave.length} mouse events`);
      }
    } catch (error) {
      console.error('Error saving mouse events:', error);
    }
  };

  const handleMouseMove = (event: MouseEvent) => {
    if (!isActive || !sessionId) return;

    const mouseEvent: MouseEvent = {
      x: event.clientX,
      y: event.clientY,
      timestamp: new Date().toISOString()
    };

    eventsBuffer.current.push(mouseEvent);

    // Save events every 5 seconds or when buffer reaches 50 events
    const now = Date.now();
    if (eventsBuffer.current.length >= 50 || now - lastSaveTime.current >= 5000) {
      const eventsToSave = [...eventsBuffer.current];
      eventsBuffer.current = [];
      lastSaveTime.current = now;
      saveEventsToDatabase(eventsToSave);
    }
  };

  const saveRemainingEvents = async () => {
    if (eventsBuffer.current.length > 0) {
      const eventsToSave = [...eventsBuffer.current];
      eventsBuffer.current = [];
      await saveEventsToDatabase(eventsToSave);
    }
  };

  useEffect(() => {
    if (!isActive || !sessionId) return;

    // Add mouse move listener
    document.addEventListener('mousemove', handleMouseMove);

    // Set up periodic save interval (every 10 seconds)
    saveInterval.current = setInterval(() => {
      if (eventsBuffer.current.length > 0) {
        const eventsToSave = [...eventsBuffer.current];
        eventsBuffer.current = [];
        lastSaveTime.current = Date.now();
        saveEventsToDatabase(eventsToSave);
      }
    }, 10000);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      if (saveInterval.current) {
        clearInterval(saveInterval.current);
      }
      // Save any remaining events when component unmounts
      if (eventsBuffer.current.length > 0) {
        saveRemainingEvents();
      }
    };
  }, [sessionId, isActive]);

  return { saveRemainingEvents };
};