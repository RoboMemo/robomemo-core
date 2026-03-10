import { useState } from 'react';
import { LanguageProvider } from '@/context/LanguageContext';
import { Navigation } from '@/components/Navigation';
import { Footer } from '@/components/Footer';
import { Home } from '@/pages/Home';
import { Dataset } from '@/pages/Dataset';
import { Products } from '@/pages/Products';

type Page = 'home' | 'dataset' | 'products';

function AppContent() {
  const [currentPage, setCurrentPage] = useState<Page>('home');

  return (
    <div className="min-h-screen bg-black">
      <Navigation currentPage={currentPage} onPageChange={setCurrentPage} />
      <main>
        {currentPage === 'home' && <Home />}
        {currentPage === 'dataset' && <Dataset />}
        {currentPage === 'products' && <Products />}
      </main>
      <Footer />
    </div>
  );
}

function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;
