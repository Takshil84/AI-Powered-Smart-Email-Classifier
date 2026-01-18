import React from 'react';
import { Box, Tabs, Tab, Typography, Chip, Container, IconButton, Tooltip } from '@mui/material';
import { 
    Bolt as BoltIcon, 
    DarkMode as DarkModeIcon, 
    LightMode as LightModeIcon 
} from '@mui/icons-material';

function Navbar({ currentTab, onTabChange, stats, darkMode, onToggleDarkMode }) {
  return (
    <Box 
        sx={{ 
            bgcolor: 'background.paper', 
            borderBottom: 1,
            borderColor: 'divider',
            position: 'sticky', top: 0, zIndex: 1200,
            boxShadow: darkMode ? 'none' : '0 4px 6px -1px rgba(0, 0, 0, 0.02)',
            transition: 'background-color 0.3s ease'
        }}
    >
        <Container 
            maxWidth="xl" 
            sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between', 
                height: { xs: 'auto', sm: 64 },
                flexWrap: { xs: 'wrap', sm: 'nowrap' },
                py: { xs: 1.5, sm: 0 },
                px: { xs: 1, sm: 2, md: 3 },
                gap: { xs: 1, sm: 0 }
            }}
        >
            
            {/* Branding */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Box sx={{ 
                    bgcolor: 'primary.main', color: '#fff', p: 0.5, borderRadius: 1.5,
                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                    <BoltIcon fontSize="small" />
                </Box>
                <Typography 
                    variant="subtitle1" 
                    fontWeight={700} 
                    color="text.primary" 
                    sx={{ 
                        letterSpacing: '-0.02em',
                        display: { xs: 'none', sm: 'block' },
                        fontSize: { sm: '0.9rem', md: '1rem' }
                    }}
                >
                    Enterprise Interpreter
                </Typography>
            </Box>
            
            {/* Navigation */}
            <Tabs 
                value={currentTab} 
                onChange={onTabChange} 
                textColor="primary" 
                indicatorColor="primary" 
                sx={{ 
                    minHeight: { xs: 48, sm: 64 },
                    order: { xs: 3, sm: 0 },
                    width: { xs: '100%', sm: 'auto' },
                    '& .MuiTab-root': { 
                        minHeight: { xs: 48, sm: 64 }, 
                        fontWeight: 600, 
                        fontSize: { xs: '0.8rem', sm: '0.9rem' }, 
                        textTransform: 'none', 
                        px: { xs: 2, sm: 3 },
                        flex: { xs: 1, sm: 'none' }
                    } 
                }}
            >
                <Tab label="Email Analysis" />
                <Tab label="Analytics" />
                <Tab label="History" />
            </Tabs>

            {/* Right Section: Stats + Dark Mode */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: { xs: 1, sm: 1.5 } }}>
                <Chip 
                    label={`Processed: ${stats.total}`} 
                    size="small" 
                    variant="outlined" 
                    sx={{ 
                        borderColor: 'divider', 
                        bgcolor: darkMode ? 'background.default' : '#f8fafc', 
                        fontWeight: 600, 
                        color: 'text.secondary',
                        fontSize: { xs: '0.7rem', sm: '0.8rem' },
                        height: { xs: 24, sm: 32 },
                        display: { xs: 'none', sm: 'flex' }
                    }} 
                />
                <Chip 
                    label={`Critical: ${stats.urgent}`} 
                    size="small" 
                    sx={{ 
                        color: 'error.main', 
                        bgcolor: darkMode ? 'rgba(239, 68, 68, 0.15)' : '#fef2f2', 
                        fontWeight: 700, 
                        border: '1px solid',
                        borderColor: darkMode ? 'rgba(239, 68, 68, 0.3)' : '#fca5a5',
                        fontSize: { xs: '0.7rem', sm: '0.8rem' },
                        height: { xs: 24, sm: 32 }
                    }} 
                />
                
                {/* Dark Mode Toggle */}
                <Tooltip title={darkMode ? 'Light Mode' : 'Dark Mode'}>
                    <IconButton 
                        onClick={onToggleDarkMode}
                        size="small"
                        sx={{ 
                            ml: 0.5,
                            color: 'text.secondary',
                            '&:hover': { bgcolor: darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.04)' }
                        }}
                    >
                        {darkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                    </IconButton>
                </Tooltip>
            </Box>

        </Container>
    </Box>
  );
}

export default Navbar;
