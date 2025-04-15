import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# Cargar datos
# Nota: Asumimos que el dataset está guardado localmente como 'ev_data.csv'
df = pd.read_csv('Electric_Vehicle_Population_Data.csv').drop(['VIN (1-10)', 'DOL Vehicle ID'], axis=1)
city_list = df['City'].value_counts().head(100).index.to_list()
df = df[(df['Model Year'].isin(range(2019, 2025))) & (df['City'].isin(city_list))]

# Limpiar datos nulos y convertir tipos si es necesario
df['Electric Range'] = df['Electric Range'].fillna(0)
df['Base MSRP'] = np.where(df['Base MSRP'] == 0.0, 40000, df['Base MSRP'])
df['Legislative District'] = df['Legislative District'].fillna(0).astype(int)
df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)

# Procesar la columna "Vehicle Location" para extraer coordenadas
def extract_coordinates(point_str):
    try:
        # Eliminar "POINT (" y ")" y dividir por espacio
        coords = point_str.replace("POINT (", "").replace(")", "").split()
        return float(coords[0]), float(coords[1])  # longitud, latitud
    except (AttributeError, IndexError, ValueError):
        return None, None

# Aplicar la función a la columna "Vehicle Location"
df[['Longitude', 'Latitude']] = df['Vehicle Location'].apply(
    lambda x: pd.Series(extract_coordinates(x))
)

# Eliminar filas sin coordenadas válidas
df = df.dropna(subset=['Longitude', 'Latitude'])

# Crear la aplicación Dash con tema Bootstrap
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.YETI]
)

app.title='Electric Vehicle Dahboard'

# Nueva paleta de colores para mejor contraste
# Colores principales
PRIMARY_COLOR = "#69b8a6"  # Verde principal (cambiado del azul)
SECONDARY_COLOR = "#FF6B6B"  # Rojo coral (mantenido)
TERTIARY_COLOR = "#FF9F1C"  # Naranja (mantenido)
QUATERNARY_COLOR = "#2E8B57"  # Verde marino (cambiado del púrpura)
BACKGROUND_COLOR = "#f5f5dc"  # Verde claro (mantenido para el fondo)
TEXT_COLOR = "#333333"

# Función para crear el mapa de ciudades con conteos
def create_city_map(df, selected_city=None):
    # Agregar conteo por ciudad
    city_counts = df.groupby('City').size().reset_index(name='count')
    
    # Calcular centros de ciudades (promedio de coordenadas de vehículos en cada ciudad)
    city_centers = df.groupby('City')[['Latitude', 'Longitude']].mean().reset_index()
    
    # Unir conteos y coordenadas
    map_data = pd.merge(city_counts, city_centers, on='City')
    
    # Destacar ciudad seleccionada
    if selected_city:
        map_data['selected'] = map_data['City'] == selected_city
        map_data['marker_size'] = np.where(map_data['selected'], 20, 10)
        map_data['marker_color'] = np.where(map_data['selected'], PRIMARY_COLOR, TERTIARY_COLOR)
    else:
        map_data['marker_size'] = 10
        map_data['marker_color'] = TERTIARY_COLOR
    
    # Crear mapa de burbujas con nueva paleta de colores
    fig = px.scatter_map(
        map_data,
        lat='Latitude',
        lon='Longitude',
        size='count',
        color='count',
        color_continuous_scale=px.colors.sequential.Cividis,
        size_max=45,
        zoom=8.0,
        center={"lat": map_data['Latitude'].mean(), "lon": map_data['Longitude'].mean()},
        map_style="carto-positron",
        hover_name='City',
        hover_data={'count': True, 'Latitude': False, 'Longitude': False},
        labels={'count': 'Número de EVs'},
        template='plotly_white'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(len=0.5, thickness=20,orientation="h", y=-0.15, x=0.5, xanchor='center', title="Number of EVs")
     
    )
    
    return fig


# Función para crear el diagrama Sankey
def create_sankey_diagram(df, selected_city=None):
    # Filtrar por ciudad si se selecciona una
    if selected_city:
        filtered_df = df[df['City'] == selected_city]
    else:
        filtered_df = df
    
    # Preparar datos para el diagrama Sankey
    # Conexiones: Marca -> Tipo de EV -> Elegibilidad CAFV
    
    # Filtrar a las 10 marcas más comunes para no sobrecargar el diagrama
    top_makes = filtered_df['Make'].value_counts().nlargest(10).index.tolist()
    filtered_df_top = filtered_df[filtered_df['Make'].isin(top_makes)]
    
    # Obtener valores únicos para nodos
    makes = filtered_df_top['Make'].unique().tolist()
    ev_types = filtered_df_top['Electric Vehicle Type'].unique().tolist()
    cafv_status = filtered_df_top['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique().tolist()
    
    # Crear lista de nodos
    nodes = makes + ev_types + cafv_status
    
    # Crear mapeo de nodos a índices
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Crear conexiones
    source = []
    target = []
    value = []
    
    # Conexiones Marca -> Tipo EV
    make_type_counts = filtered_df_top.groupby(['Make', 'Electric Vehicle Type']).size().reset_index(name='count')
    for _, row in make_type_counts.iterrows():
        source.append(node_indices[row['Make']])
        target.append(node_indices[row['Electric Vehicle Type']])
        value.append(row['count'])
    
    # Conexiones Tipo EV -> Elegibilidad CAFV
    type_cafv_counts = filtered_df_top.groupby(['Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility']).size().reset_index(name='count')
    for _, row in type_cafv_counts.iterrows():
        source.append(node_indices[row['Electric Vehicle Type']])
        target.append(node_indices[row['Clean Alternative Fuel Vehicle (CAFV) Eligibility']])
        value.append(row['count'])
    
    # Paleta de colores para contraste
    color_palette = [
        PRIMARY_COLOR, "#64B5F6", "#90CAF9", 
        SECONDARY_COLOR, "#FF8A80", "#FFCDD2",
        TERTIARY_COLOR, "#FFCC80", "#FFE0B2",
        QUATERNARY_COLOR, "#B39DDB", "#D1C4E9"
    ]
    
    node_colors = [color_palette[i % len(color_palette)] for i in range(len(nodes))]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(100, 181, 246, 0.3)"  # Azul claro translúcido
        )
    )])
    
    title = f"Electric Vehicle Flow for {selected_city}" if selected_city else "Electric Vehicle Flow in Washington"
    
    fig.update_layout(
        title_text=title,
        font_size=10,
        # height=600
    )
    
    return fig

# Función para calcular estadísticas para las tarjetas
def calculate_card_stats(df, selected_city=None):
    # Filtrar por ciudad si se selecciona una
    if selected_city:
        filtered_df = df[df['City'] == selected_city]
    else:
        filtered_df = df
    
    # Calcular estadísticas
    total_vehicles = len(filtered_df)
    
    # Marca más popular
    popular_make = filtered_df['Make'].value_counts().idxmax() if not filtered_df.empty else "N/A"
    
    # Modelo más popular
    popular_model = filtered_df['Model'].value_counts().idxmax() if not filtered_df.empty else "N/A"
    
    # Porcentaje BEV vs PHEV
    ev_type_counts = filtered_df['Electric Vehicle Type'].value_counts(normalize=True) * 100
    bev_percent = ev_type_counts.get('Battery Electric Vehicle (BEV)', 0)
    phev_percent = ev_type_counts.get('Plug-in Hybrid Electric Vehicle (PHEV)', 0)
    
    # Promedio de rango eléctrico
    avg_range = filtered_df['Electric Range'].mean() if not filtered_df.empty else 0
    
    # Promedio de precio (Base MSRP)
    avg_price = filtered_df['Base MSRP'].mean() if not filtered_df.empty else 0
    
    # Porcentaje elegibilidad CAFV
    cafv_eligible = filtered_df[filtered_df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] == 'Eligible']['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].count()
    cafv_percent = (cafv_eligible / total_vehicles * 100) if total_vehicles > 0 else 0
    
    # Compañía eléctrica más común
    popular_utility = filtered_df['Electric Utility'].value_counts().idxmax() if not filtered_df.empty else "N/A"
    
    return {
        'total_vehicles': total_vehicles,
        'popular_make': popular_make,
        'popular_model': popular_model,
        'bev_percent': bev_percent,
        'phev_percent': phev_percent,
        'avg_range': avg_range,
        'avg_price': avg_price,
        'cafv_percent': cafv_percent,
        'popular_utility': popular_utility
    }

# Crear una función para generar tarjetas de estadísticas
def create_info_card(title, value, color, icon=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className=icon, style={"fontSize": "2rem", "color": color}) if icon else None,
                ], style={"textAlign": "center", "marginBottom": "10px"}) if icon else None,
                html.H4(title, className="card-title text-center", style={"color": TEXT_COLOR, "fontSize": "1rem"}),
                html.H3(value, className="card-text text-center", style={"color": color, "fontWeight": "bold"}),
            ]),
        ]),
        className="mb-3 shadow-sm",
        style={"backgroundColor": "white", "borderRadius": "10px", "borderTop": f"4px solid {color}"}
        )
# Función para realizar análisis predictivo de la autonomía eléctrica
def perform_range_prediction(df, selected_city=None):
    # Filtrar por ciudad si se selecciona una
    if selected_city:
        filtered_df = df[df['City'] == selected_city].copy()
    else:
        filtered_df = df.copy()
    
    # Eliminar filas con valores nulos en la variable objetivo
    filtered_df = filtered_df.dropna(subset=['Electric Range'])
    
    # Si hay muy pocos datos, devolver un mensaje
    if len(filtered_df) < 50:
        return None, "Insufficient data for predictive analysis"
    
    # Seleccionar características para el modelo
    X = filtered_df[['Make', 'Model Year', 'Electric Vehicle Type']]
    y = filtered_df['Electric Range']
    
    # Asegurarnos de que solo usamos categorías con suficientes datos
    # Filtrar categorías poco comunes para evitar problemas con one-hot encoding
    make_counts = X['Make'].value_counts()
    common_makes = make_counts[make_counts >= 5].index
    X = X[X['Make'].isin(common_makes)]
    y = y.loc[X.index]  # Filtrar y para mantener las mismas filas
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocesamiento de datos
        categorical_features = ['Make', 'Electric Vehicle Type']
        numeric_features = ['Model Year']
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numeric_transformer, numeric_features)
            ])
        
        # Crear pipeline con preprocesamiento y modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Entrenar el modelo
        pipeline.fit(X_train, y_train)
        
        # Realizar predicciones
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Para el gráfico de importancia de características, usamos un enfoque alternativo
        # Entrenamos un modelo simplificado solo para obtener las importancias
        importances = model.feature_importances_
        cat_encoder = preprocessor.named_transformers_['cat']
        
        # Generar nombres de características
        categorical_names = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_names = categorical_names + numeric_features
        
        # Crear un mapeo simplificado para visualización
        simplified_names = []
        for name in feature_names:
            if name.startswith('Make_'):
                simplified_names.append('Make: ' + name.replace('Make_', ''))
            elif name.startswith('Electric Vehicle Type_'):
                simplified_names.append('Type: ' + name.replace('Electric Vehicle Type_', ''))
            else:
                simplified_names.append(name)
        
        # Ordenar por importancia y tomar los 10 principales
        feature_importance = sorted(zip(simplified_names, importances), key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in feature_importance[:6]]
        top_importances = [x[1] for x in feature_importance[:6]]
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "indicator"}, {"type": "indicator"}]  # Changed this line
            ],
            subplot_titles=["Prediction vs. Actual Value", "Feature Importance"],  # Updated titles
            column_widths=[0.5, 0.5],
            row_heights=[0.7, 0.3]
        )
        fig.update_layout(template="plotly_white")
       
        # Gráfico de dispersión: Predicción vs Valor Real
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(
                    size=8,
                    color=PRIMARY_COLOR,
                    opacity=0.7
                ),
                name='Predictions',
                hovertemplate='Actual Value: %{x:.1f} mi<br>Predicción: %{y:.1f} mi',
            ),
            row=1, col=1
        )
        
        # Línea de referencia perfecta
        max_range = max(max(y_test), max(y_pred))
        min_range = min(min(y_test), min(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_range, max_range],
                y=[min_range, max_range],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect Prediction',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Gráfico de barras: Importancia de características
        fig.add_trace(
            go.Bar(
                x=top_importances[::-1],
                y=top_features[::-1],
                orientation='h',
                marker_color=TERTIARY_COLOR,
                hovertemplate='Importance: %{x:.3f}',

            ),
            row=1, col=2
        )
        
        # Indicadores de métricas
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=r2,
                domain={'x': [0, 0.45], 'y': [0, 1]},
                title={'text': "R² Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.6], 'color': "gray"},
                        {'range': [0.6, 1], 'color': SECONDARY_COLOR}
                    ],
                    'bar': {'color': PRIMARY_COLOR}
                },
                delta={'reference': 0.75}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=mae,
                domain={'x': [0.55, 1], 'y': [0, 1]},
                title={'text': "Mean Absolute Error (miles)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 20], 'color': SECONDARY_COLOR},
                        {'range': [20, 50], 'color': "gray"},
                        {'range': [50, 100], 'color': "lightgray"}
                    ],
                    'bar': {'color': PRIMARY_COLOR}
                }
            ),
            row=2, col=2
        )
        
        # Actualizar diseño
        title_text = "Electric Range Prediction"
        if selected_city:
            title_text += f" for {selected_city}"
            
        
        fig.update_layout(
            title=title_text,
            xaxis_title="Actual Range (miles)",
            yaxis_title="Predicted Range (miles)",
            xaxis2_title="Importance",
            # height=700,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Determinar el factor más importante
        if top_features and top_importances:
            most_important_feature = top_features[0]
            
            if "Make:" in most_important_feature:
                make_name = most_important_feature.replace("Make: ", "")
                insight_message = [
                    html.P([
                        "The predictive model shows that the make ",
                        html.B(f"{make_name}", style={"color": PRIMARY_COLOR}),
                        "is the factor with the greatest influence on electric range."
                    ])
                ]
            elif "Type:" in most_important_feature:
                type_name = most_important_feature.replace("Type: ", "")
                insight_message = [
                    html.P([
                        "The predictive model shows that the vehicle type ",
                        html.B(f"{type_name}", style={"color": PRIMARY_COLOR}),
                        " is the factor with the greatest influence on electric range."
                    ])
                ]
            else:
                insight_message = [
                    html.P([
                        "The predictive model shows that ",
                        html.B(f"Model Year", style={"color": PRIMARY_COLOR}),
                        " is the factor with the greatest influence on electric range."
                    ])
                ]
            
            insight_message.extend([
                html.P([
                    "The average prediction error is ",
                    html.B(f"{mae:.1f} millas", style={"color": SECONDARY_COLOR}),
                    "."
                ]),
                html.P([
                    "The model explains ",
                    html.B(f"{r2:.0%}", style={"color": QUATERNARY_COLOR}),
                    " of the variability in electric vehicle range."
                ])
            ])
        else:
            insight_message = [
                html.P([
                    "The average prediction error is ",
                    html.B(f"{mae:.1f} millas", style={"color": SECONDARY_COLOR}),
                    "."
                ]),
                html.P([
                    "The model explains ",
                    html.B(f"{r2:.0%}", style={"color": QUATERNARY_COLOR}),
                    " of the variability in electric vehicle range"
                ])
            ]
        
        return fig, html.Div(insight_message, className="mt-3")
        
    except Exception as e:
        return None, f"Error performing predictive analysis: {str(e)}"
# Diseño del layout de la aplicación con Bootstrap y tema ecológico
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh", "padding": "20px"},
    children=[
        # Header con logo y título
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-leaf", style={"color": PRIMARY_COLOR, "fontSize": "2.5rem", "marginRight": "15px"}),
                    html.H2("WA DOL Electric Vehicle Registrations: A 2019-2024 Analysis", 
                           style={"color": PRIMARY_COLOR, "display": "inline"}
                           )
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"})
            ], width=12, className="mb-4 text-center")
        ]),
        
        # Almacenamiento del estado seleccionado
        dcc.Store(id='selected-city', data=None),
        
        # Nueva disposición: Mapa y tarjetas en la misma fila
        dbc.Row([
            # Columna para el mapa (8 columnas)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Geographic Spread of Electric Vehicles in Washington", 
                               className="text-center m-0", 
                               style={"color": PRIMARY_COLOR}),
                            dcc.Markdown("*Click on a city on the map to explore its data.*",
                         className="text-center mt-2",
                         style={"fontSize": "1.2rem", "color":PRIMARY_COLOR})
                    ], style={"backgroundColor": "white"}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='city-map',
                            figure=create_city_map(df),
                            config={'displayModeBar': True},
                            style={"height": "60vh"}
                        )
                    ])
                ], className="shadow")
            ], width=8, className="mb-4"), 
            # Columna para las tarjetas informativas (4 columnas)
            dbc.Col([
                html.H4(id="stats-title", className="text-center mb-3", style={"color": PRIMARY_COLOR}),
                
                # Contenedor con scroll para tarjetas
                html.Div(
                    id='info-cards',
                    style={
                        "overflowY": "auto",
                        "maxHeight": "60vh",
                        "paddingRight": "10px"
                    }
                ),
                
                # Botón para restablecer selección
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Reset to See All Cities"
                    ], id='reset-button', n_clicks=0, color="success", className="shadow w-100 mt-3")
                ])
            ], width=4, className="mb-4")
        ]),
        
        # Contenedor para el diagrama Sankey en una fila separada
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Tracing Electric Vehicle Characteristics in Washington", 
                               className="text-center m-0", 
                               style={"color": PRIMARY_COLOR})
                    ], style={"backgroundColor": "white"}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='sankey-diagram',
                            figure=create_sankey_diagram(df),
                            config={'displayModeBar': True},
                            style={"height": "60vh"}
                        )
                    ])
                ], className="shadow")
            ], width=12, className="mb-4")
        ]),
        
        # NUEVA SECCIÓN: Análisis de Clusters
     # NUEVA SECCIÓN: Análisis Predictivo de Autonomía
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.H4("Forecasting EV Autonomy: The Impact of Vehicle Characteristics", 
                                       className="text-center m-0", 
                                       style={"color": PRIMARY_COLOR})
                            ], width=9),
                            dbc.Col([
                                dbc.Button(
                                    "Model Details",  # Use text instead of an icon
                                    id="prediction-info-button",
                                    color="success",
                                    size="md",
                                    className="float-end"
                                )
                            ], width=3)
                        ], align="center")
                    ], style={"backgroundColor": "white"}),
                    dbc.CardBody([
                        html.Div(id='prediction-message', className="text-center"),
                        dcc.Graph(
                            id='range-prediction',
                            config={'displayModeBar': True},
                            style={"height": "60vh"}
                        )
                    ])
                ], className="shadow")
            ], width=12, className="mb-4")
        ]), 
        dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("About Predictive Analysis"), close_button=True),
            dbc.ModalBody([
                html.H5("Random Forest Regressor Model", className="mb-3", style={"color": PRIMARY_COLOR}),
                html.P([
                    "This section uses a ", 
                    html.B("machine learning model"),
                    " to predict electric vehicle range based on various characteristics."
                ]),
                html.Hr(),
                html.H6("How It Works:", className="mt-3", style={"color": QUATERNARY_COLOR}),
                html.Ul([
                    html.Li("The model analyzes patterns in existing electric vehicle data to identify relationships between vehicle attributes and range."),
                    html.Li("It combines multiple decision trees (like a 'forest' of different prediction paths) to make more accurate predictions."),
                    html.Li("Each tree makes its own prediction, and the final result is the average of all trees' predictions."),
                ]),
                html.Hr(),
                html.H6("Understanding the Results:", className="mt-3", style={"color": QUATERNARY_COLOR}),
                html.Ul([
                    html.Li([
                        html.B("Prediction vs. Actual Value: "),
                        "Shows how well the model's predictions match actual vehicle ranges. Points closer to the diagonal line indicate better predictions."
                    ]),
                    html.Li([
                        html.B("Feature Importance: "),
                        "Shows which vehicle characteristics have the greatest influence on electric range."
                    ]),
                    html.Li([
                        html.B("R² Score: "),
                        "Measures how well the model explains variations in range (higher is better, with 1.0 being perfect)."
                    ]),
                    html.Li([
                        html.B("Mean Absolute Error: "),
                        "The average prediction error in miles (lower is better)."
                    ]),
                ]),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-prediction-modal", className="ms-auto", color="success")
            ),
        ],
        id="prediction-info-modal",
        size="lg",
        is_open=False,
            )
      ]
    )
    # Callbacks para la interactividad

# Callback para capturar la ciudad seleccionada en el mapa
@app.callback(
    Output('selected-city', 'data'),
    [Input('city-map', 'clickData'),
     Input('reset-button', 'n_clicks')
    ],
    prevent_initial_call=True
)
def update_selected_city(clickData, n_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'city-map' and clickData is not None:
        # Extraer el nombre de la ciudad seleccionada
        city_name = clickData['points'][0]['hovertext']
        return city_name
    elif trigger_id == 'reset-button':
        # Restablecer la selección
        return None
    
    return None

# Callback para actualizar el mapa basado en la ciudad seleccionada
@app.callback(
    Output('city-map', 'figure'),
    Input('selected-city', 'data')
)
def update_map(selected_city):
    return create_city_map(df, selected_city)

# Callback para actualizar el diagrama Sankey basado en la ciudad seleccionada
@app.callback(
    Output('sankey-diagram', 'figure'),
    Input('selected-city', 'data')
)
def update_sankey(selected_city):
    return create_sankey_diagram(df, selected_city)

# Callback para actualizar el título de las estadísticas
@app.callback(
    Output('stats-title', 'children'),
    Input('selected-city', 'data')
)
def update_stats_title(selected_city):
    if selected_city:
        return f"Stats for {selected_city}"
    else:
        return "Statewide Statistics for Washington"

# Callback para actualizar las tarjetas informativas
@app.callback(
    Output('info-cards', 'children'),
    Input('selected-city', 'data')
)
def update_info_cards_component(selected_city):
    # Obtener estadísticas
    stats = calculate_card_stats(df, selected_city)
    
    # Crear tarjetas informativas en formato vertical
    return html.Div([
        # Primera tarjeta
        create_info_card("Total Vehicles", f"{stats['total_vehicles']:,}", PRIMARY_COLOR, "fas fa-car"),
        
        # Segunda tarjeta
        create_info_card("Most Popular Make", stats['popular_make'], SECONDARY_COLOR, "fas fa-industry"),
        
        # Tercera tarjeta
        create_info_card("Most Popular Model", stats['popular_model'], TERTIARY_COLOR, "fas fa-tag"),
        
        # Cuarta tarjeta (BEV vs PHEV)
        dbc.Card(
            dbc.CardBody([
                html.H4("BEV vs PHEV Distribution", className="card-title text-center", style={"color": TEXT_COLOR, "fontSize": "1rem"}),
                dbc.Row([
                    dbc.Col([
                        html.P("BEV", style={"fontWeight": "bold", "textAlign": "center"}),
                        html.H3(f"{stats['bev_percent']:.1f}%", style={"color": PRIMARY_COLOR, "textAlign": "center"}),
                    ], width=6),
                    dbc.Col([
                        html.P("PHEV", style={"fontWeight": "bold", "textAlign": "center"}),
                        html.H3(f"{stats['phev_percent']:.1f}%", style={"color": SECONDARY_COLOR, "textAlign": "center"}),
                    ], width=6),
                ]),
            ]),
            className="mb-3 shadow-sm",
            style={"backgroundColor": "white", "borderRadius": "10px", "borderTop": f"4px solid {QUATERNARY_COLOR}"}
        ),
        
        # Quinta tarjeta
        create_info_card("Average Electric Range", f"{stats['avg_range']:.1f} mi", QUATERNARY_COLOR, "fas fa-road"),
        
        # Sexta tarjeta
        create_info_card("Average Price", f"${stats['avg_price']:,.0f}", PRIMARY_COLOR, "fas fa-dollar-sign"),
        
        # Séptima tarjeta
        create_info_card("CAFV Elegilbility", f"{stats['cafv_percent']:.1f}%", SECONDARY_COLOR, "fas fa-leaf"),
        
        # Octava tarjeta
        create_info_card("Leading Electric Utility", stats['popular_utility'], TERTIARY_COLOR, "fas fa-bolt"),
    ])

@app.callback(
    [Output('range-prediction', 'figure'),
     Output('prediction-message', 'children')],
    Input('selected-city', 'data')
)
def update_range_prediction(selected_city):
    try:
        fig, message = perform_range_prediction(df, selected_city)
        
        if fig is None:
            # Si no hay suficientes datos, mostrar un mensaje
            return go.Figure(), html.Div([
                html.I(className="fas fa-exclamation-circle me-2", style={"color": SECONDARY_COLOR}),
                html.Span(message, style={"color": TEXT_COLOR})
            ], className="p-3")
        
        return fig, message
        
    except Exception as e:
        # En caso de error
        return go.Figure(), html.Div([
            html.I(className="fas fa-exclamation-triangle me-2", style={"color": SECONDARY_COLOR}),
            html.Span(f"Error performing predictive analysis: {str(e)}", style={"color": TEXT_COLOR})
        ], className="p-3")

@app.callback(
    Output("prediction-info-modal", "is_open"),
    [Input("prediction-info-button", "n_clicks"), Input("close-prediction-modal", "n_clicks")],
    [dash.dependencies.State("prediction-info-modal", "is_open")],
)
def toggle_prediction_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, jupyter_mode='external', port=8052)