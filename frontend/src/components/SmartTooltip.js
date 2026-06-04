import React, { useRef, useEffect } from 'react';
import { Info, HelpCircle } from 'lucide-react';
import { METRIC_EXPLANATIONS } from '../constants';
import { interpretMetric } from '../utils/helpers';

export const SmartTooltip = ({ children, className = '' }) => {
  const tipRef = useRef(null);
  const parentRef = useRef(null);
  useEffect(() => {
    const el = tipRef.current;
    const parent = parentRef.current;
    if (!el || !parent) return;
    const handleEnter = () => {
      const rect = parent.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      el.style.left = ''; el.style.right = ''; el.style.top = ''; el.style.bottom = '';
      if (rect.left + 260 > vw) { el.style.right = '0'; el.style.left = 'auto'; }
      else { el.style.left = '0'; }
      if (rect.bottom + 120 > vh) { el.style.bottom = '100%'; el.style.top = 'auto'; el.style.marginBottom = '6px'; el.style.marginTop = '0'; }
      else { el.style.top = '100%'; el.style.bottom = 'auto'; el.style.marginTop = '6px'; el.style.marginBottom = '0'; }
    };
    parent.addEventListener('mouseenter', handleEnter);
    return () => parent.removeEventListener('mouseenter', handleEnter);
  }, []);
  return (
    <span ref={parentRef} className={`group/tip relative inline-flex items-center gap-1 cursor-help ${className}`}>
      {React.Children.map(children, (child, i) =>
        i === React.Children.count(children) - 1 ? <span ref={tipRef} className="invisible group-hover/tip:visible opacity-0 group-hover/tip:opacity-100 absolute z-[9999] w-64 p-2.5 rounded-lg bg-popover border shadow-xl text-xs text-popover-foreground transition-all duration-150 pointer-events-none">{child}</span> : child
      )}
    </span>
  );
};

export const MetricTip = ({ metricKey, children, className = '', value }) => {
  const info = METRIC_EXPLANATIONS[metricKey];
  if (!info) return <span className={className}>{children}</span>;
  const interp = value !== undefined ? interpretMetric(metricKey, value) : null;
  return (
    <SmartTooltip className={className}>
      {children}
      <Info className="h-3 w-3 text-muted-foreground/50 group-hover/tip:text-foreground transition-colors shrink-0" />
      <span data-testid={`metric-tip-${metricKey}`}>
        <span className="font-semibold block mb-0.5">{info.name}</span>
        <span className="text-muted-foreground leading-relaxed block">{info.description}</span>
        {interp && <span className={`block mt-1.5 pt-1.5 border-t leading-relaxed font-medium ${interp.color}`}>{interp.text}</span>}
      </span>
    </SmartTooltip>
  );
};

export const HelpTip = ({ text, children, className = '' }) => (
  <SmartTooltip className={className}>
    {children}
    <span className="cursor-help" data-testid="help-tip-icon">
      <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/40 group-hover/tip:text-blue-500 transition-colors shrink-0" />
    </span>
    <span>{text}</span>
  </SmartTooltip>
);
