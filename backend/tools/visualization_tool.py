import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import io

# Modern, accessible color palette
COLOR_SEQUENCE = [
    "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#64748B"
]

class VisualizationTool:
    """
    Autonomous Visualization Engine.
    Generates tailored, responsive Plotly charts adhering to data visualization best practices.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_chart(
        self,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        title: Optional[str] = None,
        aggregation: str = "sum",
        top_n: int = 12
    ) -> Optional[go.Figure]:
        """Generate high-quality Plotly chart based on analytical intent."""
        fig = None
        c_type = chart_type.lower()

        try:
            if c_type in ("line", "trend", "time_series"):
                fig = self._create_line_chart(x_col, y_col, color_col, title, aggregation)
            elif c_type in ("bar", "comparison", "ranking"):
                fig = self._create_bar_chart(x_col, y_col, color_col, title, aggregation, top_n)
            elif c_type in ("scatter", "relationship"):
                fig = self._create_scatter_chart(x_col, y_col, color_col, title)
            elif c_type in ("distribution", "histogram"):
                fig = self._create_distribution_chart(x_col or y_col, color_col, title)
            elif c_type in ("box", "boxplot"):
                fig = self._create_box_chart(x_col, y_col, color_col, title)
            elif c_type in ("pie", "donut", "composition"):
                fig = self._create_pie_chart(x_col, y_col, title, top_n)
            elif c_type in ("heatmap", "correlation"):
                fig = self._create_correlation_heatmap(title)
            else:
                # Default heuristic
                if x_col and y_col:
                    fig = self._create_bar_chart(x_col, y_col, color_col, title, aggregation, top_n)
                elif x_col or y_col:
                    fig = self._create_distribution_chart(x_col or y_col, color_col, title)

            if fig:
                self._apply_theme(fig, title or f"{chart_type.title()} Chart")
            return fig
        except Exception as e:
            # Fallback error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart generation note: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#EF4444")
            )
            self._apply_theme(fig, title or "Analysis Chart")
            return fig

    def _apply_theme(self, fig: go.Figure, title: str):
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, color="#1E293B", family="Inter, sans-serif"),
                x=0.02
            ),
            colorway=COLOR_SEQUENCE,
            margin=dict(l=40, r=40, t=60, b=40),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

    def _create_line_chart(self, x_col: str, y_col: str, color_col: Optional[str], title: str, agg: str) -> go.Figure:
        clean = self.df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy()
        
        # Check if x is date
        if pd.api.types.is_datetime64_any_dtype(clean[x_col]) or 'date' in x_col.lower():
            clean[x_col] = pd.to_datetime(clean[x_col], errors='coerce')
            clean = clean.dropna(subset=[x_col]).sort_values(x_col)

        group_cols = [x_col] + ([color_col] if color_col else [])
        if agg == "mean":
            grouped = clean.groupby(group_cols, as_index=False)[y_col].mean()
        elif agg == "count":
            grouped = clean.groupby(group_cols, as_index=False)[y_col].count()
        else:
            grouped = clean.groupby(group_cols, as_index=False)[y_col].sum()

        fig = px.line(
            grouped,
            x=x_col,
            y=y_col,
            color=color_col,
            markers=True,
            title=title
        )
        fig.update_traces(line=dict(width=2.5))
        return fig

    def _create_bar_chart(self, x_col: str, y_col: str, color_col: Optional[str], title: str, agg: str, top_n: int) -> go.Figure:
        clean = self.df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy()
        group_cols = [x_col] + ([color_col] if color_col else [])
        
        if agg == "mean":
            grouped = clean.groupby(group_cols, as_index=False)[y_col].mean()
        elif agg == "count":
            grouped = clean.groupby(group_cols, as_index=False)[y_col].count()
        else:
            grouped = clean.groupby(group_cols, as_index=False)[y_col].sum()

        # Sort descending by metric
        grouped = grouped.sort_values(by=y_col, ascending=False).head(top_n)

        # Use horizontal bar if labels are long or more than 8 categories
        is_long = any(len(str(val)) > 14 for val in grouped[x_col]) or len(grouped) > 8
        if is_long:
            grouped = grouped.sort_values(by=y_col, ascending=True)  # ascending so top appears on top in horizontal
            fig = px.bar(
                grouped,
                y=x_col,
                x=y_col,
                color=color_col,
                orientation='h',
                title=title,
                text_auto='.2s'
            )
        else:
            fig = px.bar(
                grouped,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                text_auto='.2s'
            )
        return fig

    def _create_scatter_chart(self, x_col: str, y_col: str, color_col: Optional[str], title: str) -> go.Figure:
        clean = self.df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy()
        # Cap scatter at 1500 points for performance
        if len(clean) > 1500:
            clean = clean.sample(n=1500, random_state=42)

        fig = px.scatter(
            clean,
            x=x_col,
            y=y_col,
            color=color_col,
            trendline="ols" if not color_col and len(clean) > 10 else None,
            title=title,
            opacity=0.75
        )
        return fig

    def _create_distribution_chart(self, col: str, color_col: Optional[str], title: str) -> go.Figure:
        clean = self.df[[col] + ([color_col] if color_col else [])].dropna().copy()
        fig = px.histogram(
            clean,
            x=col,
            color=color_col,
            marginal="box",
            nbins=35,
            title=title
        )
        return fig

    def _create_box_chart(self, x_col: Optional[str], y_col: str, color_col: Optional[str], title: str) -> go.Figure:
        clean = self.df[([x_col] if x_col else []) + [y_col] + ([color_col] if color_col else [])].dropna().copy()
        fig = px.box(
            clean,
            x=x_col,
            y=y_col,
            color=color_col,
            points="outliers",
            title=title
        )
        return fig

    def _create_pie_chart(self, cat_col: str, val_col: Optional[str], title: str, top_n: int) -> go.Figure:
        clean = self.df[[cat_col] + ([val_col] if val_col else [])].dropna().copy()
        if val_col:
            grouped = clean.groupby(cat_col, as_index=False)[val_col].sum()
        else:
            grouped = clean[cat_col].value_counts().reset_index()
            grouped.columns = [cat_col, "count"]
            val_col = "count"

        grouped = grouped.sort_values(by=val_col, ascending=False)
        if len(grouped) > top_n:
            top_part = grouped.head(top_n - 1)
            other_val = grouped.iloc[top_n - 1:][val_col].sum()
            other_row = pd.DataFrame([{cat_col: "Other", val_col: other_val}])
            grouped = pd.concat([top_part, other_row], ignore_index=True)

        fig = px.pie(
            grouped,
            names=cat_col,
            values=val_col,
            hole=0.45,
            title=title
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def _create_correlation_heatmap(self, title: str) -> go.Figure:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr = self.df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=title
        )
        return fig
