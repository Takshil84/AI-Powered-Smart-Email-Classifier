import React, { useState, useEffect, useMemo } from 'react';
import { ThemeProvider, createTheme, Box, CssBaseline, Snackbar, Alert } from '@mui/material';
import Navbar from './components/Navbar';
import Analysis from './pages/Analysis';
import Analytics from './pages/Analytics';
import History from './pages/History';
import axios from 'axios';
import { API_URL } from './config';  // ✅ Import backend URL

// -- Theme Generator (same as before) --
const getTheme = (mode) => createTheme({
  palette: { /* ... existing theme code ... */ },
  typography: { /* ... existing typography code ... */ },
  components: { /* ... existing components overrides ... */ }
});

function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [toast, setToast] = useState({ open: false, message: '', severity: 'success' });
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [subject, setSubject] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);

  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('email_focus_v1');
      return saved ? JSON.parse(saved) : [];
    } catch (e) { return []; }
  });

  useEffect(() => { localStorage.setItem('darkMode', JSON.stringify(darkMode)); }, [darkMode]);
  useEffect(() => { localStorage.setItem('email_focus_v1', JSON.stringify(history)); }, [history]);
  const theme = useMemo(() => getTheme(darkMode ? 'dark' : 'light'), [darkMode]);

  const showToast = (message, severity = 'success') => setToast({ open: true, message, severity });
  const hideToast = () => setToast(prev => ({ ...prev, open: false }));

  // ✅ Updated handleClassify using API_URL
  const handleClassify = async () => {
    if (!content.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/api/classify`, { subject, content });
      const result = {
        id: Date.now(),
        subject,
        text: content,
        ...response.data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setLastResult(result);
      setHistory(prev => [result, ...prev]);
      
      if (result.confidence < confidenceThreshold) {
        showToast(`Low confidence result (${(result.confidence * 100).toFixed(0)}%)`, 'warning');
      } else if (result.urgency === 'High') {
        showToast('⚠️ High urgency email detected!', 'error');
      } else {
        showToast('Email analyzed successfully', 'success');
      }
    } catch (err) {
      setError("System Unreachable. Is backend running?");
      showToast('Analysis failed. Check backend connection.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => { setHistory([]); showToast('History cleared', 'info'); };
  const deleteHistoryItem = (id) => setHistory(prev => prev.filter(item => item.id !== id));
  const getUrgencyColor = (u) => {
    switch((u || "").toLowerCase()) {
      case 'high': return theme.palette.error.main;
      case 'medium': return theme.palette.warning.main;
      case 'low': return theme.palette.success.main;
      default: return theme.palette.text.secondary;
    }
  };

  const handleTabChange = (event, newValue) => setActiveTab(newValue);
  const toggleDarkMode = () => setDarkMode(prev => !prev);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', pb: 8, transition: 'background-color 0.3s ease' }}>
        <Navbar 
            currentTab={activeTab} 
            onTabChange={handleTabChange}
            stats={{ total: history.length, urgent: history.filter(h => (h.urgency||'').toLowerCase()==='high').length }}
            darkMode={darkMode}
            onToggleDarkMode={toggleDarkMode}
        />
        {activeTab===0 && <Analysis 
            subject={subject} setSubject={setSubject}
            content={content} setContent={setContent}
            loading={loading} error={error}
            handleClassify={handleClassify}
            lastResult={lastResult} setLastResult={setLastResult}
            getUrgencyColor={getUrgencyColor}
            confidenceThreshold={confidenceThreshold}
        />}
        {activeTab===1 && <Analytics history={history} />}
        {activeTab===2 && <History 
            history={history} 
            clearHistory={clearHistory} 
            deleteHistoryItem={deleteHistoryItem}
            getUrgencyColor={getUrgencyColor} 
        />}
        <Snackbar 
          open={toast.open} 
          autoHideDuration={4000} 
          onClose={hideToast}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={hideToast} severity={toast.severity} variant="filled" sx={{ width: '100%' }}>
            {toast.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
