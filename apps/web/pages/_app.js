import "@/styles/globals.css";
import { StrictMode } from 'react';

export default function App({ Component, pageProps }) {
  // Disable StrictMode in development to prevent double execution
  if (process.env.NODE_ENV === 'development') {
    return <Component {...pageProps} />;
  }
  
  return (
    <StrictMode>
      <Component {...pageProps} />
    </StrictMode>
  );
}
