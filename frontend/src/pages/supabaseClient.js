import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://appqqpxjhiehvcgtivvs.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwcHFxcHhqaGllaHZjZ3RpdnZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTk5ODQ4MzYsImV4cCI6MjAzNTU2MDgzNn0._jdG1rPRq6J62pYxJX-XYGirQhOXUSVmLaNi7l8y9rQ';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
