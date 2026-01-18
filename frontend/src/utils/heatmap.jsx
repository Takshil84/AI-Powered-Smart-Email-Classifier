import React from 'react';
import { Tooltip as MuiTooltip } from '@mui/material';

export const renderHeatmap = (text, scores) => {
    if (!scores?.length) return text;
    return text.split(' ').map((word, i) => {
        const clean = word.replace(/[^\w]/g,"");
        const match = scores.find(s => s.word === word || s.word === clean);
        if (match) {
            return (
                <MuiTooltip key={i} title={`Signal Strength: ${(match.score*100).toFixed(0)}%`}>
                    <span style={{ 
                        backgroundColor: `rgba(239, 68, 68, ${Math.min(match.score*3, 0.5)})`, 
                        fontWeight: 700, padding: '0 2px', borderRadius: 4, cursor: 'help',
                        borderBottom: '2px solid rgba(239, 68, 68, 0.4)'
                    }}>{word} </span>
                </MuiTooltip>
            );
        }
        return word + ' ';
    });
};
