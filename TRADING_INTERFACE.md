# PRATE Trading & Admin Interface

## Overview

This is a comprehensive SPA-like web interface for the PRATE (Prime-Resonant Adaptive Trading Ecology) system. It provides a professional trading interface with real-time market data, system administration tools, and monitoring capabilities.

## Features

### Trading Interface
- **TradingView Chart Integration**: Professional-grade charting with full technical analysis capabilities
- **Live Order Book**: Real-time order book data from Binance WebSocket feed
- **Trading Panel**: Quick order entry with support for Limit, Market, and Stop orders
- **Position Management**: Track open positions and account balances

### Admin Dashboard
- **System Configuration**: Configure trading mode, strategies, and position limits
- **API Management**: Set up exchange connections and API credentials
- **Guild System**: Enable/disable trading guilds and monitor performance
- **Database Management**: Monitor candle storage and database health

### Monitoring
- **Performance Metrics**: Track P&L, Sharpe ratio, win rate, and drawdown
- **System Health**: Monitor CPU, memory, and network usage
- **Trade Activity**: View recent trades and activity statistics
- **System Logs**: Real-time log viewer for debugging

### Risk Management
- **Risk Limits**: Configure max daily loss, position size, and leverage
- **KAM Protection**: KAM (Kill All Machines) safety system monitoring
- **Portfolio Analysis**: VaR, Expected Shortfall, and Beta calculations

### Memory System
- **Holographic Memory**: Monitor HRR memory utilization and patterns
- **Entropy & Phase**: Track entropy levels and phase coherence
- **Prime Basis**: View active prime basis selection
- **Context Retrieval**: Monitor memory retrieval performance

### Backtesting
- **Configuration**: Set up backtest parameters and date ranges
- **Results Display**: View performance metrics and statistics
- **Equity Curve**: Visualize strategy performance over time

## Usage

### Quick Start

1. Open `trading_admin.html` in a modern web browser (Chrome, Firefox, Edge, or Safari)
2. The interface will automatically connect to Binance's public WebSocket for live market data
3. Navigate between sections using the left sidebar menu

### Navigation

- **Trading**: Main trading interface with chart and order entry
- **Admin**: System configuration and API management
- **Monitoring**: Performance metrics and system logs
- **Risk**: Risk management and portfolio analysis
- **Memory**: Holographic memory system visualization
- **Backtest**: Backtesting configuration and results

### Requirements

- Modern web browser with JavaScript enabled
- Internet connection for:
  - TailwindCSS (styling)
  - Feather Icons (UI icons)
  - TradingView widgets (charts)
  - Binance WebSocket (live data)

## Technical Details

### Architecture

The interface is built as a single-page application (SPA) with:
- **Tab-based navigation**: All content loaded on initial page load, switched dynamically
- **Real-time data**: WebSocket connection to Binance for live order book updates
- **Responsive design**: Flexbox-based layout that adapts to different screen sizes
- **No backend required**: Runs entirely in the browser (for demo purposes)

### Components

1. **Order Book Component**
   - Adapted from `in/ob.html` and optimized for compact display
   - Shows top 10 bids and asks with cumulative volume visualization
   - Updates in real-time via WebSocket

2. **TradingView Chart**
   - Embedded TradingView widget with dark theme
   - Full technical analysis tools
   - 15-minute timeframe by default

3. **Tab System**
   - JavaScript-based tab switching
   - Active tab highlighting
   - Lazy rendering for performance

### Customization

You can customize the interface by editing `trading_admin.html`:

- **Colors**: Modify the Tailwind color classes (e.g., `bg-gray-800`, `text-blue-400`)
- **Layout**: Adjust grid columns and flexbox properties
- **Data Source**: Change the WebSocket URL to connect to different exchanges
- **Chart Settings**: Modify TradingView widget configuration

## Integration with PRATE Backend

This is a frontend interface. To integrate with the PRATE Python backend:

1. Create a FastAPI or Flask backend server
2. Implement WebSocket endpoints for real-time updates
3. Add REST API endpoints for:
   - Order execution
   - Configuration updates
   - Metrics retrieval
   - System status
4. Update the JavaScript in `trading_admin.html` to connect to your backend

Example WebSocket integration:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Update UI with backend data
};
```

## Security Notes

⚠️ **IMPORTANT**: This is a demo interface with no authentication or encryption.

Before using in production:
- Implement proper authentication (OAuth, JWT, etc.)
- Use HTTPS/WSS for all connections
- Never expose API keys in frontend code
- Implement rate limiting and input validation
- Add CSRF protection for all mutations

## Future Enhancements

- [ ] FastAPI backend integration
- [ ] User authentication and sessions
- [ ] Real-time WebSocket updates from PRATE backend
- [ ] Order history and trade analytics
- [ ] Strategy backtesting visualization
- [ ] Multi-symbol watchlist
- [ ] Alert notifications
- [ ] Mobile-responsive improvements

## License

Copyright (c) 2024. All rights reserved.

This software is for educational and research purposes only. Trading involves substantial risk of loss.
