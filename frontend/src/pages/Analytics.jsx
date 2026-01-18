import React, { useMemo } from 'react';
import { 
    Box, Container, Paper, Typography, Grid, Button
} from '@mui/material';
import { 
    PieChart, Pie, Cell, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { 
    Download as DownloadIcon,
    Analytics as AnalyticsIcon,
    TrendingUp as TrendingIcon,
    Category as CategoryIcon,
    Speed as SpeedIcon
} from '@mui/icons-material';
import { exportAnalyticsPDF } from '../utils/pdfExport';

// Color palette
const URGENCY_COLORS = {
    High: '#ef4444',
    Medium: '#f59e0b', 
    Low: '#10b981'
};

// Fixed colors for categories
const CATEGORY_COLORS = [
    '#6366f1', '#8b5cf6', '#ec4899', '#14b8a6', 
    '#f97316', '#06b6d4', '#84cc16', '#f43f5e'
];

function Analytics({ history }) {
    // Compute statistics from history
    const stats = useMemo(() => {
        if (!history || history.length === 0) {
            return {
                total: 0,
                urgencyBreakdown: { High: 0, Medium: 0, Low: 0 },
                categoryBreakdown: {},
                dailyVolume: [],
                confidenceDistribution: [],
                avgConfidence: 0,
                urgencyData: [],
                categoryData: []
            };
        }

        // Urgency breakdown
        const urgencyBreakdown = { High: 0, Medium: 0, Low: 0 };
        const categoryBreakdown = {};
        let totalConfidence = 0;
        let confidenceCount = 0;
        const dailyMap = {};

        history.forEach(item => {
            // Urgency - Normalize Case
            let urgency = item.urgency;
            if (urgency) {
                // Capitalize first letter, rest lowercase to ensure "High"/"high" matches "High"
                urgency = urgency.charAt(0).toUpperCase() + urgency.slice(1).toLowerCase();
                if (urgencyBreakdown[urgency] !== undefined) {
                    urgencyBreakdown[urgency]++;
                } else {
                    // Fallback for unexpected values
                    urgencyBreakdown.Low = (urgencyBreakdown.Low || 0) + 1; 
                }
            }
            
            // Category
            if (item.category) {
                categoryBreakdown[item.category] = (categoryBreakdown[item.category] || 0) + 1;
            }
            
            // Confidence - Check for non-null/undefined (allow 0)
            if (item.confidence !== undefined && item.confidence !== null) {
                totalConfidence += item.confidence;
                confidenceCount++;
            }
            
            // Daily volume - Use item timestamp
            try {
                // item.timestamp is coming from toLocaleString()
                const dateObj = new Date(item.timestamp);
                if (!isNaN(dateObj)) {
                    const dateStr = dateObj.toLocaleDateString();
                    dailyMap[dateStr] = (dailyMap[dateStr] || 0) + 1;
                }
            } catch (e) {
                console.warn("Invalid date:", item.timestamp);
            }
        });

        // Convert to chart data
        const urgencyData = Object.entries(urgencyBreakdown)
            .filter(([_, v]) => v > 0)
            .map(([name, value]) => ({ name, value, color: URGENCY_COLORS[name] || '#94a3b8' }));

        const categoryData = Object.entries(categoryBreakdown)
            .map(([name, value]) => ({ 
                name: name.length > 15 ? name.substring(0, 15) + '...' : name, 
                value,
                fullName: name 
            }))
            .sort((a, b) => b.value - a.value) // Sort by count descending
            .slice(0, 8); // Top 8 categories

        const dailyVolume = Object.entries(dailyMap)
            .map(([date, count]) => ({ date, count }))
            .sort((a, b) => new Date(a.date) - new Date(b.date))
            .slice(-7); // Last 7 days

        // Confidence distribution (buckets)
        const confidenceBuckets = { '< 50%': 0, '50-70%': 0, '70-85%': 0, '85-95%': 0, '> 95%': 0 };
        history.forEach(item => {
            if (item.confidence !== undefined && item.confidence !== null) {
                const conf = item.confidence * 100;
                if (conf < 50) confidenceBuckets['< 50%']++;
                else if (conf < 70) confidenceBuckets['50-70%']++;
                else if (conf < 85) confidenceBuckets['70-85%']++;
                else if (conf < 95) confidenceBuckets['85-95%']++;
                else confidenceBuckets['> 95%']++;
            }
        });
        const confidenceDistribution = Object.entries(confidenceBuckets)
            .map(([range, count]) => ({ range, count }));

        return {
            total: history.length,
            urgencyBreakdown,
            categoryBreakdown,
            urgencyData,
            categoryData,
            dailyVolume,
            confidenceDistribution,
            avgConfidence: confidenceCount > 0 ? totalConfidence / confidenceCount : 0
        };
    }, [history]);

    const handleExport = () => {
        exportAnalyticsPDF(stats);
    };

    // Empty state
    if (!history || history.length === 0) {
        return (
            <Container 
                maxWidth={false} 
                disableGutters
                sx={{ 
                    mt: { xs: 2, sm: 4 }, 
                    height: { xs: 'auto', md: 'calc(100vh - 120px)' }, 
                    pb: 2,
                    px: { xs: 2, sm: 3, md: 3 },
                    width: '100%'
                }}
            >
                <Box sx={{ 
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 4, 
                    textAlign: 'center', 
                    bgcolor: 'background.paper', 
                    borderRadius: 3,
                    border: '1px dashed',
                    borderColor: 'divider'
                }}>
                    <Box>
                        <AnalyticsIcon sx={{ fontSize: { xs: 48, sm: 64 }, color: 'text.disabled', mb: 2 }} />
                        <Typography variant="h6" color="text.primary" gutterBottom>No Analytics Data</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Analyze some emails first to see insights and charts here.
                        </Typography>
                    </Box>
                </Box>
            </Container>
        );
    }

    return (
        <Container 
            maxWidth={false} 
            disableGutters
            sx={{ 
                mt: { xs: 2, sm: 4 }, 
                height: { xs: 'auto', md: 'calc(100vh - 120px)' }, 
                pb: 2,
                px: { xs: 2, sm: 3, md: 3 },
                width: '100%',
                overflow: 'hidden'
            }}
        >
            <Box 
                sx={{ 
                    display: 'flex',
                    flexDirection: { xs: 'column', md: 'row' },
                    gap: 3,
                    height: '100%',
                    width: '100%'
                }}
            >
                {/* LEFT PANEL: STATS & CONTROLS */}
                <Box 
                    sx={{ 
                        width: { xs: '100%', md: '280px', lg: '320px' },
                        flexShrink: 0,
                        height: { xs: 'auto', md: '100%' },
                        display: 'flex',
                        flexDirection: 'column'
                    }}
                >
                    <Paper 
                        sx={{ 
                            p: 3, 
                            height: '100%', 
                            display: 'flex', 
                            flexDirection: 'column', 
                            bgcolor: 'background.paper',
                            overflowY: 'auto',
                            borderRadius: 3
                        }}
                        elevation={0}
                        variant="outlined"
                    >
                        {/* Header */}
                        <Box sx={{ mb: 4 }}>
                            <Typography variant="h5" fontWeight={700} gutterBottom>
                                Analytics
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Overview of {stats.total} analyzed emails.
                            </Typography>
                            <Button 
                                startIcon={<DownloadIcon />} 
                                variant="outlined"
                                onClick={handleExport}
                                size="small"
                                fullWidth
                                sx={{ justifyContent: 'flex-start', mt: 1 }}
                            >
                                Export Report
                            </Button>
                        </Box>

                        <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2, fontWeight: 700, letterSpacing: 0.5 }}>
                            KEY METRICS
                        </Typography>

                        {/* Vertical Stats Stack */}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {[
                                { label: 'Total Analyzed', value: stats.total, icon: <AnalyticsIcon />, color: '#6366f1' },
                                { label: 'High Urgency', value: stats.urgencyBreakdown.High, icon: <TrendingIcon />, color: '#ef4444' },
                                { label: 'Categories', value: Object.keys(stats.categoryBreakdown).length, icon: <CategoryIcon />, color: '#8b5cf6' },
                                { label: 'Avg Confidence', value: `${(stats.avgConfidence * 100).toFixed(0)}%`, icon: <SpeedIcon />, color: '#10b981' }
                            ].map((stat, idx) => (
                                <Box key={idx} sx={{ 
                                    p: 2, 
                                    borderRadius: 3, 
                                    bgcolor: 'background.default',
                                    // border: '1px solid',
                                    // borderColor: 'divider',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 2
                                }}>
                                    <Box sx={{ 
                                        p: 1.25, 
                                        borderRadius: 2, 
                                        bgcolor: `${stat.color}15`,
                                        color: stat.color,
                                        display: 'flex'
                                    }}>
                                        {React.cloneElement(stat.icon, { fontSize: 'medium' })}
                                    </Box>
                                    <Box>
                                        <Typography variant="h6" fontWeight={700} sx={{ lineHeight: 1 }}>
                                            {stat.value}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                                            {stat.label}
                                        </Typography>
                                    </Box>
                                </Box>
                            ))}
                        </Box>
                    </Paper>
                </Box>

                {/* RIGHT PANEL: CHARTS */}
                <Box 
                    sx={{ 
                        flex: 1, 
                        height: { xs: 'auto', md: '100%' }, 
                        minWidth: 0, // Prevent flex overflow
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 3
                    }}
                >
                    {/* Top Row: Urgency & Confidence */}
                    <Box sx={{ 
                        flex: 1, 
                        display: 'flex', 
                        flexDirection: { xs: 'column', lg: 'row' }, 
                        gap: 3, 
                        minHeight: 0 
                    }}>
                             {/* Urgency */}
                             <Paper sx={{ 
                                flex: 1, 
                                p: 3, 
                                display: 'flex', 
                                flexDirection: 'column',
                                minHeight: 300,
                                borderRadius: 3
                            }} elevation={0} variant="outlined">
                                <Typography variant="subtitle1" fontWeight={600} gutterBottom>Urgency Distribution</Typography>
                                <Box sx={{ flex: 1, minHeight: 0 }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={stats.urgencyData}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius="60%"
                                                outerRadius="85%"
                                                paddingAngle={3}
                                                dataKey="value"
                                                labelLine={false}
                                                label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
                                                    if (percent < 0.05) return null;
                                                    const RADIAN = Math.PI / 180;
                                                    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
                                                    const x = cx + radius * Math.cos(-midAngle * RADIAN);
                                                    const y = cy + radius * Math.sin(-midAngle * RADIAN);
                                                    return (
                                                      <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={700}>
                                                        {`${(percent * 100).toFixed(0)}%`}
                                                      </text>
                                                    );
                                                  }}
                                            >
                                                {stats.urgencyData.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <Tooltip />
                                            <Legend verticalAlign="bottom" height={36}/>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>

                            {/* Confidence */}
                            <Paper sx={{ 
                                flex: 1, 
                                p: 2, 
                                display: 'flex',
                                flexDirection: 'column',
                                minHeight: 300
                            }}>
                                <Typography variant="subtitle1" fontWeight={600} gutterBottom>Confidence Scores</Typography>
                                <Box sx={{ flex: 1, minHeight: 0 }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={stats.confidenceDistribution} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                            <XAxis 
                                                dataKey="range" 
                                                tick={{ fontSize: 10 }} 
                                                interval={0} 
                                                angle={-15}
                                                textAnchor="end"
                                                height={50}
                                            />
                                            <YAxis tick={{ fontSize: 11 }} />
                                            <Tooltip cursor={{fill: 'transparent'}} />
                                            <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={32} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </Box>
                            </Paper>
                        </Box>

                        {/* Bottom Row: Category (Needs width) */}
                        <Paper sx={{ 
                            flex: 1, 
                            p: 2, 
                            display: 'flex', 
                            flexDirection: 'column',
                            minHeight: 300
                        }}>
                             <Typography variant="subtitle1" fontWeight={600} gutterBottom>Category Breakdown</Typography>
                             <Box sx={{ flex: 1, minHeight: 0 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                     <BarChart data={stats.categoryData} layout="vertical" margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                                         <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                         <XAxis type="number" hide />
                                         <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                                         <Tooltip 
                                             cursor={{fill: 'transparent'}}
                                             formatter={(value, name, props) => {
                                                 if (props && props.payload && props.payload.fullName) return [value, props.payload.fullName];
                                                 return [value, name];
                                             }}
                                         />
                                         <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                             {stats.categoryData.map((entry, index) => (
                                                 <Cell key={`cell-${index}`} fill={CATEGORY_COLORS[index % CATEGORY_COLORS.length]} />
                                             ))}
                                         </Bar>
                                     </BarChart>
                                 </ResponsiveContainer>
                             </Box>
                        </Paper>
                    </Box>
                </Box>

        </Container>
    );
}

export default Analytics;
