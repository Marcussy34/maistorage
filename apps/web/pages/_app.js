import "../styles/globals.css";
import { StrictMode } from 'react';
import { Baloo_2 } from 'next/font/google';

// Configure Baloo 2 font with optimal settings
const baloo2 = Baloo_2({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-baloo2',
});

export default function App({ Component, pageProps }) {
  // Disable StrictMode in development to prevent double execution
  if (process.env.NODE_ENV === 'development') {
    return (
      <main className={`${baloo2.className} ${baloo2.variable}`}>
        <Component {...pageProps} />
      </main>
    );
  }
  
  return (
    <StrictMode>
      <main className={`${baloo2.className} ${baloo2.variable}`}>
        <Component {...pageProps} />
      </main>
    </StrictMode>
  );
}
