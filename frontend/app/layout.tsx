import type { Metadata } from 'next';
import './style.css';

export const metadata: Metadata = {
  title: 'BCI Oscillatory Encoder | Research Workbench',
  description: 'Computational visual-to-oscillatory encoding and signal inspection.',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
