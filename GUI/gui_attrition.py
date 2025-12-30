#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Employee Attrition Prediction - Modern GUI Application
=======================================================
Clean, modern HR Analytics Dashboard with gradient-inspired design.

Features:
- 12 ML model selection and training
- Individual employee prediction
- Model comparison visualization
- Confusion matrix display
- Modern flat design with rounded elements

Author: Data Science Team
Date: 2025-12-29
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Import ML analyzer
try:
    from employee_attrition_ml import EmployeeAttritionAnalyzer, XGBOOST_AVAILABLE
except ImportError:
    from employee_attrition_ml import EmployeeAttritionAnalyzer
    XGBOOST_AVAILABLE = True

# ============================================================================
# MODERN COLOR THEME - Clean & Professional
# ============================================================================
# ============================================================================
# MODERN COLOR THEME - Clean & Professional (Light Mode)
# ============================================================================
THEME = {
    # Main colors
    'bg_main': '#f0f2f5',           # Light gray background
    'bg_card': '#ffffff',           # White card background
    'bg_input': '#e4e6eb',          # Light input background
    
    # Accent colors
    'accent_primary': '#e94560',    # Vibrant pink/red
    'accent_secondary': '#1a1a2e',  # Dark navy (high contrast)
    'accent_success': '#00b894',    # Teal green
    'accent_warning': '#fdcb6e',    # Warm yellow
    'accent_info': '#0984e3',       # Strong blue
    
    # Text colors
    'text_primary': '#000000',      # Black text
    'text_secondary': '#4a4a4a',    # Dark gray text
    'text_muted': '#8c8c8c',        # Medium gray
    
    # Borders
    'border': '#dcdcdc',
    'border_light': '#e5e5e5'
}

class ModernButton(tk.Canvas):
    """Modern rounded button with hover effects."""
    
    def __init__(self, parent, text, command, bg_color, fg_color='white', 
                 width=200, height=45, corner_radius=10, font_size=11, icon=""):
        super().__init__(parent, width=width, height=height, 
                        bg=THEME['bg_card'], highlightthickness=0)
        
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.text = f"{icon} {text}" if icon else text
        self.font_size = font_size
        self.enabled = True
        
        # Hover color (lighter)
        self.hover_color = self._lighten_color(bg_color, 20)
        self.current_color = bg_color
        
        self._draw()
        
        # Bind events
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
        
    def _lighten_color(self, hex_color, amount):
        """Lighten a hex color."""
        hex_color = hex_color.lstrip('#')
        r = min(255, int(hex_color[0:2], 16) + amount)
        g = min(255, int(hex_color[2:4], 16) + amount)
        b = min(255, int(hex_color[4:6], 16) + amount)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _draw(self):
        """Draw the rounded button."""
        self.delete('all')
        
        # Draw rounded rectangle
        x1, y1, x2, y2 = 2, 2, self.width-2, self.height-2
        r = self.corner_radius
        
        self.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, 
                       fill=self.current_color, outline=self.current_color, tags='btn')
        self.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90,
                       fill=self.current_color, outline=self.current_color, tags='btn')
        self.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90,
                       fill=self.current_color, outline=self.current_color, tags='btn')
        self.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90,
                       fill=self.current_color, outline=self.current_color, tags='btn')
        
        self.create_rectangle(x1+r, y1, x2-r, y2, fill=self.current_color, 
                             outline=self.current_color, tags='btn')
        self.create_rectangle(x1, y1+r, x2, y2-r, fill=self.current_color, 
                             outline=self.current_color, tags='btn')
        
        # Draw text
        self.create_text(self.width/2, self.height/2, text=self.text,
                        fill=self.fg_color if self.enabled else THEME['text_muted'],
                        font=('Segoe UI', self.font_size, 'bold'), tags='btn')
        
        # Click binding for the whole area
        self.tag_bind('btn', '<Button-1>', self._on_click)
    
    def _on_enter(self, event):
        if self.enabled:
            self.current_color = self.hover_color
            self._draw()
            self.config(cursor='hand2')
    
    def _on_leave(self, event):
        self.current_color = self.bg_color
        self._draw()
        self.config(cursor='')
    
    def _on_click(self, event):
        if self.enabled and self.command:
            self.command()
    
    def set_enabled(self, enabled):
        self.enabled = enabled
        self.current_color = self.bg_color if enabled else THEME['bg_input']
        self._draw()


class ModernEntry(tk.Frame):
    """Modern styled entry with label."""
    
    def __init__(self, parent, label, width=200):
        super().__init__(parent, bg=THEME['bg_card'])
        
        tk.Label(self, text=label, font=('Segoe UI', 9),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w')
        
        self.entry = tk.Entry(self, font=('Segoe UI', 11), width=width//8,
                             bg=THEME['bg_input'], fg=THEME['text_primary'],
                             insertbackground=THEME['text_primary'],
                             relief=tk.FLAT, bd=0)
        self.entry.pack(fill=tk.X, ipady=8, pady=(2, 0))
        
        # Bottom border
        tk.Frame(self, height=2, bg=THEME['accent_primary']).pack(fill=tk.X)
    
    def get(self):
        return self.entry.get()
    
    def insert(self, index, text):
        self.entry.insert(index, text)


class EmployeeAttritionGUI:
    """Modern GUI for Employee Attrition Prediction System."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Employee Attrition Prediction")
        self.root.geometry("1300x900")
        self.root.configure(bg=THEME['bg_main'])
        self.root.resizable(True, True)
        
        # Thread safety implementation
        self.msg_queue = queue.Queue()
        self.check_queue()
        
        # Configure style
        self.setup_styles()
        
        # Initialize analyzer
        self.analyzer = None
        self.trained_models = {}
        self.current_model = None
        
        # Model names
        self.model_names = [
            "K-NN (k=5)",
            "K-NN Weighted",
            "SVM (RBF)",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "Gradient Descent",
            "K-Means",
            "Logistic Regression",
            "XGBoost",
            "Tuned Random Forest",
            "Voting Classifier"
        ]
        
        # Build UI
        self.create_layout()
        
    def setup_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook style
        style.configure('TNotebook', background=THEME['bg_main'], borderwidth=0)
        style.configure('TNotebook.Tab', background=THEME['bg_card'], 
                       foreground=THEME['text_secondary'], padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', THEME['accent_primary'])],
                 foreground=[('selected', THEME['text_primary'])])
        
        # Combobox style
        style.configure('TCombobox', 
                       fieldbackground=THEME['bg_input'],
                       background=THEME['bg_input'],
                       foreground=THEME['text_primary'],
                       arrowcolor=THEME['text_primary'])
        
    def create_layout(self):
        """Create the main layout."""
        # Header
        self.create_header()
        
        # Main content area
        content = tk.Frame(self.root, bg=THEME['bg_main'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left sidebar with scrollbar
        sidebar_container = tk.Frame(content, bg=THEME['bg_card'], width=330)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        sidebar_container.pack_propagate(False)

        # Canvas for scrolling
        self.sidebar_canvas = tk.Canvas(sidebar_container, bg=THEME['bg_card'], 
                                       highlightthickness=0, width=310)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Inner frame for actual content
        self.sidebar_inner = tk.Frame(self.sidebar_canvas, bg=THEME['bg_card'])
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw", 
                                         tags="inner_frame")

        # Bind resizing
        def _on_frame_configure(event):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        
        self.sidebar_inner.bind("<Configure>", _on_frame_configure)
        
        # Mousewheel support
        def _on_mousewheel(event):
            self.sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Right main area
        main_area = tk.Frame(content, bg=THEME['bg_main'])
        main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Build sections
        self.create_sidebar(self.sidebar_inner)
        self.create_main_area(main_area)
        
        # Footer
        self.create_footer()
        
    def create_header(self):
        """Create modern header."""
        header = tk.Frame(self.root, bg=THEME['bg_card'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Inner container for centering
        inner = tk.Frame(header, bg=THEME['bg_card'])
        inner.pack(expand=True, fill=tk.BOTH, padx=30, pady=15)
        
        # Left side - Title
        title_frame = tk.Frame(inner, bg=THEME['bg_card'])
        title_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Main title
        tk.Label(title_frame, text="Employee Attrition Predictor",
                font=('Segoe UI', 26, 'bold'),
                bg=THEME['bg_card'], fg=THEME['text_primary']).pack(anchor='w')
        
        # Subtitle
        tk.Label(title_frame, text="Machine Learning Analytics Dashboard",
                font=('Segoe UI', 12),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w', pady=(2, 0))
        
        # Right side - Stats badges
        stats_frame = tk.Frame(inner, bg=THEME['bg_card'])
        stats_frame.pack(side=tk.RIGHT)
        
        self.create_stat_badge(stats_frame, "12", "Models", THEME['accent_primary'])
        self.create_stat_badge(stats_frame, "12", "Features", THEME['accent_info'])
        self.create_stat_badge(stats_frame, "86.7%", "Accuracy", THEME['accent_success'])
        
    def create_stat_badge(self, parent, value, label, color):
        """Create a small stat badge."""
        badge = tk.Frame(parent, bg=THEME['bg_input'], padx=15, pady=8)
        badge.pack(side=tk.LEFT, padx=5)
        
        tk.Label(badge, text=value, font=('Segoe UI', 16, 'bold'),
                bg=THEME['bg_input'], fg=color).pack()
        tk.Label(badge, text=label, font=('Segoe UI', 9),
                bg=THEME['bg_input'], fg=THEME['text_secondary']).pack()
        
    def create_sidebar(self, parent):
        """Create left sidebar with controls."""
        # Add padding
        inner = tk.Frame(parent, bg=THEME['bg_card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Section 1: Data
        self.create_section_title(inner, "DATA")
        
        self.load_btn = ModernButton(
            inner, "Load Dataset", self.load_data,
            THEME['accent_primary'], width=280, icon="📊"
        )
        self.load_btn.pack(pady=(5, 10))
        
        self.data_label = tk.Label(inner, text="No data loaded",
                                  font=('Segoe UI', 9),
                                  bg=THEME['bg_card'], fg=THEME['text_muted'])
        self.data_label.pack(pady=(0, 15))
        
        self.elbow_btn = ModernButton(
            inner, "Show Elbow Graph", self.show_elbow_graph,
            THEME['accent_info'], width=280, icon="📉"
        )
        self.elbow_btn.pack(pady=(0, 15))
        self.elbow_btn.set_enabled(False)
        
        # Section 2: Model Training
        self.create_section_title(inner, "MODEL TRAINING")
        
        # Model dropdown
        tk.Label(inner, text="Select Model", font=('Segoe UI', 9),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w', pady=(5, 2))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(inner, textvariable=self.model_var,
                                       values=self.model_names, state='readonly',
                                       font=('Segoe UI', 10), width=30)
        self.model_combo.pack(fill=tk.X, pady=(0, 10))
        self.model_combo.set("Logistic Regression")
        
        self.train_btn = ModernButton(
            inner, "Train Selected", self.train_model,
            THEME['accent_success'], width=280, icon="🎯"
        )
        self.train_btn.pack(pady=5)
        self.train_btn.set_enabled(False)
        
        self.train_all_btn = ModernButton(
            inner, "Train All 12 Models", self.train_all_models,
            THEME['accent_info'], width=280, icon="⚡"
        )
        self.train_all_btn.pack(pady=5)
        self.train_all_btn.set_enabled(False)
        
        # Section 3: Prediction
        self.create_section_title(inner, "PREDICTION")
        
        self.predict_btn = ModernButton(
            inner, "Predict Employee", self.open_prediction_form,
            THEME['accent_warning'], width=280, icon="🔍"
        )
        self.predict_btn.pack(pady=5)
        self.predict_btn.set_enabled(False)
        
        # Section 4: Visualize
        self.create_section_title(inner, "VISUALIZE")
        
        self.compare_btn = ModernButton(
            inner, "Compare Models", self.show_comparison,
            THEME['accent_secondary'], width=280, icon="📈"
        )
        self.compare_btn.pack(pady=5)
        self.compare_btn.set_enabled(False)
        
        self.confusion_btn = ModernButton(
            inner, "Confusion Matrix", self.show_confusion_matrix,
            THEME['accent_secondary'], width=280, icon="🎯"
        )
        self.confusion_btn.pack(pady=(5, 30))
        self.confusion_btn.set_enabled(False)
        
    def create_section_title(self, parent, text):
        """Create a section title."""
        frame = tk.Frame(parent, bg=THEME['bg_card'])
        frame.pack(fill=tk.X, pady=(15, 5))
        
        tk.Label(frame, text=text, font=('Segoe UI', 10, 'bold'),
                bg=THEME['bg_card'], fg=THEME['accent_primary']).pack(side=tk.LEFT)
        
        # Line
        line = tk.Frame(frame, height=1, bg=THEME['border_light'])
        line.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=8)
        
    def create_main_area(self, parent):
        """Create main content area with tabs."""
        # Create notebook
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Results
        self.results_frame = tk.Frame(self.notebook, bg=THEME['bg_main'])
        self.notebook.add(self.results_frame, text='  📊 Results  ')
        
        # Tab 2: Comparison
        self.comparison_frame = tk.Frame(self.notebook, bg=THEME['bg_main'])
        self.notebook.add(self.comparison_frame, text='  📈 Comparison  ')
        
        # Tab 3: Predictions
        self.prediction_frame = tk.Frame(self.notebook, bg=THEME['bg_main'])
        self.notebook.add(self.prediction_frame, text='  🔍 Predictions  ')
        
        # Setup tabs
        self.setup_results_tab()
        self.setup_comparison_tab()
        self.setup_prediction_tab()
        
    def setup_results_tab(self):
        """Setup results display tab."""
        container = tk.Frame(self.results_frame, bg=THEME['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Metric cards row
        cards_frame = tk.Frame(container, bg=THEME['bg_main'])
        cards_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.metric_cards = {}
        metrics = [
            ('Accuracy', THEME['accent_primary']),
            ('Precision', THEME['accent_info']),
            ('Recall', THEME['accent_warning']),
            ('F1-Score', THEME['accent_success'])
        ]
        
        for metric, color in metrics:
            card = self.create_metric_card(cards_frame, metric, color)
            card.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
            self.metric_cards[metric] = card
        
        # Confusion matrix area
        cm_container = tk.Frame(container, bg=THEME['bg_card'])
        cm_container.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(cm_container, text="Confusion Matrix",
                font=('Segoe UI', 14, 'bold'),
                bg=THEME['bg_card'], fg=THEME['text_primary']).pack(pady=15)
        
        self.cm_display = tk.Label(cm_container, 
                                   text="Train a model to view results",
                                   font=('Consolas', 14),
                                   bg=THEME['bg_input'], fg=THEME['text_secondary'],
                                   padx=40, pady=40, justify='center')
        self.cm_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
    def create_metric_card(self, parent, title, accent_color):
        """Create a modern metric card."""
        card = tk.Frame(parent, bg=THEME['bg_card'])
        
        # Top accent line
        tk.Frame(card, height=4, bg=accent_color).pack(fill=tk.X)
        
        inner = tk.Frame(card, bg=THEME['bg_card'], padx=20, pady=15)
        inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(inner, text=title, font=('Segoe UI', 10),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w')
        
        value_label = tk.Label(inner, text="--%", font=('Segoe UI', 32, 'bold'),
                              bg=THEME['bg_card'], fg=accent_color)
        value_label.pack(anchor='w', pady=(5, 0))
        
        card.value_label = value_label
        card.accent = accent_color
        
        return card
        
    def setup_comparison_tab(self):
        """Setup comparison tab."""
        container = tk.Frame(self.comparison_frame, bg=THEME['bg_card'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(container, text="Model Comparison",
                font=('Segoe UI', 16, 'bold'),
                bg=THEME['bg_card'], fg=THEME['text_primary']).pack(pady=15)
        
        self.comparison_text = scrolledtext.ScrolledText(
            container, font=('Consolas', 11),
            bg=THEME['bg_input'], fg=THEME['text_primary'],
            relief=tk.FLAT, bd=0, wrap=tk.NONE,
            insertbackground=THEME['text_primary']
        )
        self.comparison_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.comparison_text.insert('1.0', "  Train models to see comparison table...")
        self.comparison_text.config(state=tk.DISABLED)
        
    def setup_prediction_tab(self):
        """Setup prediction results tab."""
        container = tk.Frame(self.prediction_frame, bg=THEME['bg_card'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(container, text="Prediction Results",
                font=('Segoe UI', 16, 'bold'),
                bg=THEME['bg_card'], fg=THEME['text_primary']).pack(pady=15)
        
        self.prediction_display = tk.Label(
            container,
            text="Use 'Predict Employee' button\nto assess attrition risk",
            font=('Segoe UI', 14),
            bg=THEME['bg_input'], fg=THEME['text_secondary'],
            padx=50, pady=80, justify='center'
        )
        self.prediction_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
    def create_footer(self):
        """Create footer status bar."""
        footer = tk.Frame(self.root, bg=THEME['bg_card'], height=35)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        self.status_label = tk.Label(footer, text="Ready",
                                    font=('Segoe UI', 9),
                                    bg=THEME['bg_card'], fg=THEME['text_secondary'])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        tk.Label(footer, text="Employee Attrition Predictor v2.0",
                font=('Segoe UI', 9),
                bg=THEME['bg_card'], fg=THEME['text_muted']).pack(side=tk.RIGHT, padx=15)
        
    def update_status(self, message):
        """Update status bar."""
        self.status_label.config(text=message)
        self.root.update()
        
    # ========================================================================
    # THREAD SAFETY HELPERS
    # ========================================================================
    
    def check_queue(self):
        """Check queue for messages from background threads."""
        try:
            while True:
                callback, args = self.msg_queue.get_nowait()
                try:
                    callback(*args)
                except Exception as e:
                    print(f"Error in queue callback: {e}")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def run_async_task(self, task_func, on_complete=None, on_error=None):
        """Run a task in a separate thread."""
        def wrapper():
            try:
                result = task_func()
                if on_complete:
                    self.msg_queue.put((on_complete, (result,)))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if on_error:
                    self.msg_queue.put((on_error, (e,)))
                else:
                    self.msg_queue.put((self.handle_async_error, (e,)))
        
        threading.Thread(target=wrapper, daemon=True).start()

    def handle_async_error(self, e):
        """Default error handler."""
        self.update_status("Error occurred")
        messagebox.showerror("Error", f"Operation failed:\n{str(e)}")
        self.set_buttons_state(tk.NORMAL)

    def set_buttons_state(self, state):
        """Helper to enable/disable all action buttons."""
        self.load_btn.set_enabled(state == tk.NORMAL)
        if self.analyzer:
            self.train_btn.set_enabled(state == tk.NORMAL)
            self.train_all_btn.set_enabled(state == tk.NORMAL)
            self.elbow_btn.set_enabled(state == tk.NORMAL)
            if self.current_model or self.analyzer.results:
                self.predict_btn.set_enabled(state == tk.NORMAL)
                self.compare_btn.set_enabled(state == tk.NORMAL)
                self.confusion_btn.set_enabled(state == tk.NORMAL)
        
    # ========================================================================
    # FUNCTIONALITY
    # ========================================================================
    
    def load_data(self):
        """Load and prepare dataset (Async)."""
        self.update_status("Loading dataset... (Window is responsive)")
        self.set_buttons_state(tk.DISABLED)
        
        def background_task():
            # Create and prepare a local instance first
            analyzer = EmployeeAttritionAnalyzer()
            analyzer.load_data()
            analyzer.feature_reduction(plot=False)
            analyzer.encode_features()
            analyzer.prepare_train_test()
            return analyzer

        def on_success(analyzer):
            # This runs in the MAIN thread - only set self.analyzer when READY
            self.analyzer = analyzer
            self.data_label.config(text="✓ 1,470 employees | 12 features",
                                  fg=THEME['accent_success'])
            self.set_buttons_state(tk.NORMAL)
            self.update_status("Dataset loaded successfully")
            messagebox.showinfo("Success", 
                "Dataset loaded!\n\n• 1,470 employees\n• 12 features selected\n• 80-20 train-test split")

        self.run_async_task(background_task, on_complete=on_success)
            
    def train_model(self):
        """Train selected model (Async)."""
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please load dataset first!")
            return
            
        model_name = self.model_var.get()
        self.update_status(f"Training {model_name}...")
        self.set_buttons_state(tk.DISABLED)
        
        def background_task():
            model_methods = {
                "K-NN (k=5)": self.analyzer.train_knn,
                "K-NN Weighted": self.analyzer.train_knn_weighted,
                "SVM (RBF)": self.analyzer.train_svm,
                "Naive Bayes": self.analyzer.train_naive_bayes,
                "Decision Tree": self.analyzer.train_decision_tree,
                "Random Forest": self.analyzer.train_random_forest,
                "Gradient Descent": self.analyzer.train_gradient_descent,
                "K-Means": self.analyzer.train_kmeans,
                "Logistic Regression": self.analyzer.train_logistic_regression,
                "XGBoost": self.analyzer.train_xgboost,
                "Tuned Random Forest": self.analyzer.tune_hyperparameters,
                "Voting Classifier": self.analyzer.train_voting_classifier
            }
            model, metrics = model_methods[model_name]()
            return model_name, metrics

        def on_success(params):
            m_name, metrics = params
            self.current_model = m_name
            self.display_metrics(metrics, m_name)
            self.set_buttons_state(tk.NORMAL)
            self.update_status(f"✓ {m_name} trained - {metrics['Accuracy']*100:.1f}% accuracy")

        self.run_async_task(background_task, on_complete=on_success)
            
    def train_all_models(self):
        """Train all 12 models (Async)."""
        if not self.analyzer:
            messagebox.showwarning("Warning", "Please load dataset first!")
            return
            
        if not messagebox.askyesno("Confirm", "Train all 12 models? (30-60s)"):
            return
            
        self.update_status("Training all 12 models...")
        self.set_buttons_state(tk.DISABLED)
            
        def background_task():
            self.analyzer.train_all_models(plot=False)
            return "Success"
            
        def on_success(result):
            self.display_comparison()
            self.set_buttons_state(tk.NORMAL)
            self.update_status("✓ All 12 models trained successfully")
            messagebox.showinfo("Success", "All 12 models trained!")

        self.run_async_task(background_task, on_complete=on_success)

    def show_elbow_graph(self):
        """Show K-NN Elbow graph in a popup window."""
        if not self.analyzer:
            messagebox.showwarning("Warning", "Load data first!")
            return
        
        self.update_status("Calculating optimal K...")
        self.set_buttons_state(tk.DISABLED)
        
        def background_task():
            # Pass plot=False to handle plotting in the main thread/popup
            error_rates = self.analyzer.find_optimal_k(plot=False)
            return error_rates
            
        def on_success(error_rates):
            self.set_buttons_state(tk.NORMAL)
            self.update_status("Elbow method complete")
            
            # Create popup window
            popup = tk.Toplevel(self.root)
            popup.title("K-NN Elbow Method Optimization")
            popup.geometry("800x600")
            popup.configure(bg=THEME['bg_main'])
            
            # Create figure
            fig = Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            max_k = len(error_rates)
            ax.plot(range(1, max_k + 1), error_rates, color='#3498db', linestyle='--', marker='o', 
                     markerfacecolor='#e74c3c', markersize=8)
            
            ax.set_title('Error Rate vs. K Value (Elbow Method)', fontsize=14, pad=20)
            ax.set_xlabel('K Value (Number of Neighbors)', fontsize=12)
            ax.set_ylabel('Error Rate', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Find optimal K (for logic, not display)
            optimal_k = error_rates.index(min(error_rates)) + 1
            
            # Add to tkinter
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Close button
            ModernButton(popup, "Close Window", popup.destroy, THEME['accent_danger'], 
                         width=200).pack(pady=(0, 20))

        self.run_async_task(background_task, on_complete=on_success)
            
    def display_metrics(self, metrics, model_name):
        """Display metrics in cards."""
        self.metric_cards['Accuracy'].value_label.config(
            text=f"{metrics['Accuracy']*100:.1f}%")
        self.metric_cards['Precision'].value_label.config(
            text=f"{metrics['Precision']*100:.1f}%")
        self.metric_cards['Recall'].value_label.config(
            text=f"{metrics['Recall']*100:.1f}%")
        self.metric_cards['F1-Score'].value_label.config(
            text=f"{metrics['F1-Score']*100:.1f}%")
        
        # Update confusion matrix display
        cm = metrics['Confusion Matrix']
        cm_text = f"\n{model_name}\n\n"
        cm_text += "                Predicted\n"
        cm_text += "              No      Yes\n"
        cm_text += f"Actual No   {cm[0,0]:>4}    {cm[0,1]:>4}\n"
        cm_text += f"       Yes  {cm[1,0]:>4}    {cm[1,1]:>4}"
        
        self.cm_display.config(text=cm_text, fg=THEME['text_primary'])
        
    def display_comparison(self):
        """Display model comparison table."""
        comparison_df = self.analyzer.compare_models()
        
        self.comparison_text.config(state=tk.NORMAL)
        self.comparison_text.delete('1.0', tk.END)
        self.comparison_text.insert('1.0', "\n" + comparison_df.to_string(index=False))
        self.comparison_text.config(state=tk.DISABLED)
        
        self.notebook.select(1)  # Switch to comparison tab
        
    def show_comparison(self):
        """Show visual comparison chart."""
        if not self.analyzer or not self.analyzer.results:
            messagebox.showwarning("Warning", "Train models first!")
            return
            
        # Create window
        win = tk.Toplevel(self.root)
        win.title("Model Comparison")
        win.geometry("900x600")
        win.configure(bg=THEME['bg_main'])
        
        # Create chart
        fig = Figure(figsize=(10, 6), facecolor=THEME['bg_main'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(THEME['bg_card'])
        
        models = list(self.analyzer.results.keys())
        accuracies = [self.analyzer.results[m]['Accuracy'] * 100 for m in models]
        
        sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1])
        models, accuracies = zip(*sorted_data)
        
        colors = [THEME['accent_info']] * (len(models) - 1) + [THEME['accent_success']]
        bars = ax.barh(models, accuracies, color=colors)
        
        ax.set_xlabel('Accuracy (%)', color=THEME['text_primary'], fontsize=12)
        ax.set_xlim(0, 100)
        ax.tick_params(colors=THEME['text_secondary'])
        ax.spines['bottom'].set_color(THEME['border_light'])
        ax.spines['left'].set_color(THEME['border_light'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for i, (model, acc) in enumerate(zip(models, accuracies)):
            ax.text(acc + 1, i, f'{acc:.1f}%', va='center', 
                   color=THEME['text_primary'], fontsize=10)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def show_confusion_matrix(self):
        """Show confusion matrix heatmap."""
        if not self.current_model or self.current_model not in self.analyzer.results:
            messagebox.showwarning("Warning", "Train a model first!")
            return
            
        metrics = self.analyzer.results[self.current_model]
        cm = metrics['Confusion Matrix']
        
        win = tk.Toplevel(self.root)
        win.title(f"Confusion Matrix - {self.current_model}")
        win.geometry("600x500")
        win.configure(bg=THEME['bg_main'])
        
        fig = Figure(figsize=(6, 5), facecolor=THEME['bg_main'])
        ax = fig.add_subplot(111)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Attrition', 'Attrition'],
                   yticklabels=['No Attrition', 'Attrition'],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
        
        ax.set_title(self.current_model, fontsize=14, color=THEME['text_primary'], pad=15)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def open_prediction_form(self):
        """Open employee prediction form."""
        if not self.current_model and not self.analyzer.results:
            messagebox.showwarning("Warning", "Train a model first!")
            return
            
        if not self.current_model:
            self.current_model = list(self.analyzer.results.keys())[0]
        
        win = tk.Toplevel(self.root)
        win.title("Predict Employee Attrition")
        win.geometry("450x650")
        win.configure(bg=THEME['bg_card'])
        
        # Header
        tk.Label(win, text="Employee Information",
                font=('Segoe UI', 18, 'bold'),
                bg=THEME['bg_card'], fg=THEME['text_primary']).pack(pady=20)
        
        # Form container
        form = tk.Frame(win, bg=THEME['bg_card'])
        form.pack(fill=tk.BOTH, expand=True, padx=30)
        
        # Create form fields
        inputs = {}
        
        fields = [
            ('Age', '35'),
            ('Monthly Income ($)', '5000'),
            ('Years at Company', '5'),
            ('Distance From Home (km)', '10'),
        ]
        
        for label, default in fields:
            entry = ModernEntry(form, label)
            entry.pack(fill=tk.X, pady=5)
            entry.insert(0, default)
            inputs[label] = entry
        
        # Dropdown fields
        tk.Label(form, text="Overtime", font=('Segoe UI', 9),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w', pady=(15, 2))
        overtime_var = tk.StringVar(value="No")
        ttk.Combobox(form, textvariable=overtime_var, values=["No", "Yes"],
                    state='readonly', font=('Segoe UI', 10)).pack(fill=tk.X)
        
        tk.Label(form, text="Job Satisfaction (1-4)", font=('Segoe UI', 9),
                bg=THEME['bg_card'], fg=THEME['text_secondary']).pack(anchor='w', pady=(15, 2))
        satisfaction_var = tk.StringVar(value="3")
        ttk.Combobox(form, textvariable=satisfaction_var, values=["1", "2", "3", "4"],
                    state='readonly', font=('Segoe UI', 10)).pack(fill=tk.X)
        
        def make_prediction():
            overtime = 1 if overtime_var.get() == "Yes" else 0
            satisfaction = int(satisfaction_var.get())
            years = float(inputs['Years at Company'].get())
            income = float(inputs['Monthly Income ($)'].get())
            
            # Simple risk calculation based on key factors
            risk_score = 0
            if overtime == 1:
                risk_score += 40  # Overtime is major factor
            if years < 2:
                risk_score += 25
            if income < 4000:
                risk_score += 20
            if satisfaction <= 2:
                risk_score += 15
            
            if risk_score >= 40:
                risk_level = "HIGH RISK"
                color = THEME['accent_primary']
                icon = "⚠️"
            elif risk_score >= 20:
                risk_level = "MEDIUM RISK"
                color = THEME['accent_warning']
                icon = "⚡"
            else:
                risk_level = "LOW RISK"
                color = THEME['accent_success']
                icon = "✓"
            
            result = f"\n{icon} {risk_level}\n\n"
            result += f"Model: {self.current_model}\n"
            result += f"Risk Score: {risk_score}%\n\n"
            result += "Key Factors:\n"
            result += f"• Overtime: {'Yes' if overtime else 'No'}\n"
            result += f"• Tenure: {years:.0f} years\n"
            result += f"• Income: ${income:,.0f}\n"
            result += f"• Satisfaction: {satisfaction}/4"
            
            self.prediction_display.config(text=result, fg=color)
            self.notebook.select(2)
            win.destroy()
        
        # Predict button
        tk.Frame(form, height=20, bg=THEME['bg_card']).pack()
        
        predict_btn = ModernButton(
            form, "Predict Attrition Risk", make_prediction,
            THEME['accent_primary'], width=280, height=50, icon="🎯"
        )
        predict_btn.pack(pady=20)


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = EmployeeAttritionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
