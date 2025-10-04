"""
Dynamic Risk Management System with Kelly Criterion and Real-Time Monitoring
Advanced risk management for cryptocurrency trading with portfolio heat monitoring.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from collections import defaultdict, deque
import threading
import time

from ..utils.logger import system_logger, audit_logger, performance_logger


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size


@dataclass
class RiskMetrics:
    """Real-time risk metrics for the portfolio."""
    total_exposure: float
    portfolio_heat: float  # Percentage of capital at risk
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    total_pnl: float
    winning_trades: int
    losing_trades: int
    win_rate: float


class KellyCriterionCalculator:
    """Calculate optimal position sizes using Kelly Criterion."""
    
    def __init__(self):
        self.win_rate_history = deque(maxlen=100)
        self.return_history = deque(maxlen=100)
        
    def calculate_kelly_fraction(self, 
                               win_probability: float,
                               avg_win: float,
                               avg_loss: float,
                               confidence_adjustment: float = 1.0) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            win_probability: Probability of winning trade
            avg_win: Average winning return
            avg_loss: Average losing return (positive value)
            confidence_adjustment: Adjustment factor based on model confidence
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_probability, q = 1-p
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply confidence adjustment
        adjusted_kelly = kelly_fraction * confidence_adjustment
        
        # Cap at reasonable maximum (25% of capital)
        return max(0.0, min(0.25, adjusted_kelly))
    
    def update_statistics(self, is_win: bool, return_pct: float):
        """Update win rate and return statistics."""
        self.win_rate_history.append(1.0 if is_win else 0.0)
        self.return_history.append(abs(return_pct))
    
    def get_historical_kelly(self) -> float:
        """Calculate Kelly fraction based on historical performance."""
        if len(self.win_rate_history) < 10:
            return 0.02  # Conservative default
            
        win_rate = np.mean(self.win_rate_history)
        
        # Separate wins and losses
        wins = [r for i, r in enumerate(self.return_history) if self.win_rate_history[i] > 0.5]
        losses = [r for i, r in enumerate(self.return_history) if self.win_rate_history[i] < 0.5]
        
        if not wins or not losses:
            return 0.02
            
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)


class VolatilityAdjustmentModule:
    """Adjust position sizes based on market volatility."""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.price_history = defaultdict(lambda: deque(maxlen=lookback_period))
        
    def add_price(self, symbol: str, price: float):
        """Add price data for volatility calculation."""
        self.price_history[symbol].append(price)
        
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate realized volatility for a symbol."""
        prices = list(self.price_history[symbol])
        
        if len(prices) < 2:
            return 0.02  # Default 2% volatility
            
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
            
        if not returns:
            return 0.02
            
        # Annualized volatility (assuming daily data)
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(365)
        
        return max(0.001, annualized_vol)  # Minimum 0.1% volatility
    
    def get_volatility_adjustment(self, symbol: str, base_volatility: float = 0.02) -> float:
        """Get volatility adjustment factor for position sizing."""
        current_vol = self.calculate_volatility(symbol)
        
        # Inverse relationship: higher volatility = smaller positions
        volatility_ratio = base_volatility / current_vol
        
        # Cap adjustment between 0.1x and 2.0x
        return max(0.1, min(2.0, volatility_ratio))


class CircuitBreaker:
    """Circuit breaker system for extreme market conditions."""
    
    def __init__(self, 
                 daily_loss_limit: float = 0.05,  # 5% daily loss limit
                 drawdown_limit: float = 0.15,    # 15% max drawdown
                 volatility_threshold: float = 0.10):  # 10% volatility threshold
        
        self.daily_loss_limit = daily_loss_limit
        self.drawdown_limit = drawdown_limit
        self.volatility_threshold = volatility_threshold
        
        self.is_triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        
        self.daily_pnl = 0.0
        self.session_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
    def check_circuit_breaker(self, 
                            current_drawdown: float,
                            daily_pnl: float,
                            market_volatility: float) -> bool:
        """
        Check if circuit breaker should be triggered.
        
        Returns:
            True if trading should be halted
        """
        # Reset daily P&L at start of new day
        current_time = datetime.now()
        if current_time.date() > self.session_start_time.date():
            self.daily_pnl = 0.0
            self.session_start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            self.is_triggered = False  # Reset daily circuit breaker
            
        # Update daily P&L
        self.daily_pnl = daily_pnl
        
        # Check conditions
        if abs(daily_pnl) > self.daily_loss_limit:
            self._trigger_circuit_breaker(f"Daily loss limit exceeded: {daily_pnl:.3%}")
            return True
            
        if current_drawdown > self.drawdown_limit:
            self._trigger_circuit_breaker(f"Drawdown limit exceeded: {current_drawdown:.3%}")
            return True
            
        if market_volatility > self.volatility_threshold:
            self._trigger_circuit_breaker(f"Market volatility too high: {market_volatility:.3%}")
            return True
            
        return False
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker with specified reason."""
        if not self.is_triggered:
            self.is_triggered = True
            self.trigger_reason = reason
            self.trigger_time = datetime.now()
            
            system_logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
            audit_logger.critical(f"Trading halted due to circuit breaker: {reason}")
    
    def reset_circuit_breaker(self, manual_override: bool = False):
        """Reset circuit breaker (manual override or automatic)."""
        if manual_override or self._can_auto_reset():
            self.is_triggered = False
            self.trigger_reason = ""
            self.trigger_time = None
            
            system_logger.info("Circuit breaker reset")
            audit_logger.info("Trading resumed after circuit breaker reset")
    
    def _can_auto_reset(self) -> bool:
        """Check if circuit breaker can be automatically reset."""
        if not self.trigger_time:
            return False
            
        # Auto-reset after 1 hour for volatility-based triggers
        if "volatility" in self.trigger_reason.lower():
            return datetime.now() - self.trigger_time > timedelta(hours=1)
            
        # Manual reset required for loss-based triggers
        return False


class DynamicRiskManager:
    """
    Dynamic Risk Management System with Real-Time Portfolio Monitoring
    
    Features:
    - Kelly Criterion position sizing with confidence adjustment
    - Real-time volatility adjustment
    - Portfolio heat monitoring (max 10% exposure)
    - Circuit breakers for extreme conditions
    - Dynamic stop-loss and take-profit levels
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.02,  # 2% max per position
                 max_portfolio_heat: float = 0.10,  # 10% max total exposure
                 base_stop_loss: float = 0.02):     # 2% stop loss
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_portfolio_heat = max_portfolio_heat
        self.base_stop_loss = base_stop_loss
        
        # Portfolio tracking
        self.positions = {}  # symbol -> Position
        self.closed_positions = []
        self.portfolio_value_history = deque(maxlen=1000)
        self.daily_returns = deque(maxlen=252)  # 1 year of daily returns
        
        # Risk calculation modules
        self.kelly_calculator = KellyCriterionCalculator()
        self.volatility_adjuster = VolatilityAdjustmentModule()
        self.circuit_breaker = CircuitBreaker()
        
        # Performance tracking
        self.peak_portfolio_value = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        system_logger.info(f"Risk Manager initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, 
                              symbol: str,
                              signal_confidence: float,
                              predicted_return: float,
                              current_price: float) -> float:
        """
        Calculate optimal position size using multiple risk factors.
        
        Args:
            symbol: Trading symbol
            signal_confidence: Model confidence (0-1)
            predicted_return: Expected return (decimal)
            current_price: Current market price
            
        Returns:
            Position size in base currency units
        """
        # Get current portfolio metrics
        current_exposure = self.get_current_exposure()
        portfolio_heat = current_exposure / self.current_capital
        
        # Check portfolio heat limit
        remaining_heat = max(0, self.max_portfolio_heat - portfolio_heat)
        if remaining_heat < 0.01:  # Less than 1% remaining capacity
            system_logger.warning("Portfolio heat limit reached, skipping trade")
            return 0.0
        
        # Kelly Criterion calculation
        historical_kelly = self.kelly_calculator.get_historical_kelly()
        
        # Confidence-adjusted Kelly
        confidence_multiplier = 0.5 + (1.5 * signal_confidence)  # 0.5x to 2.0x
        adjusted_kelly = historical_kelly * confidence_multiplier
        
        # Volatility adjustment
        volatility_adjustment = self.volatility_adjuster.get_volatility_adjustment(symbol)
        
        # Base position size from risk parameters
        base_risk = min(self.max_position_size, adjusted_kelly)
        volatility_adjusted_risk = base_risk * volatility_adjustment
        
        # Heat-adjusted risk (don't exceed remaining portfolio heat capacity)
        heat_adjusted_risk = min(volatility_adjusted_risk, remaining_heat)
        
        # Convert to position size in base currency
        risk_amount = self.current_capital * heat_adjusted_risk
        position_size = risk_amount / current_price
        
        # Final safety checks
        min_position = self.current_capital * 0.001 / current_price  # Minimum 0.1% position
        max_position = self.current_capital * 0.05 / current_price   # Maximum 5% position
        
        final_position_size = max(min_position, min(position_size, max_position))
        
        performance_logger.info(
            f"Position sizing for {symbol}: "
            f"confidence={signal_confidence:.3f}, "
            f"kelly={adjusted_kelly:.3f}, "
            f"vol_adj={volatility_adjustment:.3f}, "
            f"size=${final_position_size * current_price:,.2f}"
        )
        
        return final_position_size
    
    def calculate_stop_loss_take_profit(self, 
                                      symbol: str,
                                      entry_price: float,
                                      side: str,
                                      confidence: float) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels.
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Get current volatility
        volatility = self.volatility_adjuster.calculate_volatility(symbol)
        
        # Confidence-adjusted stop loss (tighter stops for high confidence)
        confidence_factor = 0.5 + (0.5 * confidence)  # 0.5x to 1.0x
        stop_distance = max(self.base_stop_loss * confidence_factor, volatility * 1.5)
        
        # Risk-reward ratio based on confidence
        risk_reward_ratio = 1.5 + (1.5 * confidence)  # 1.5:1 to 3:1
        take_profit_distance = stop_distance * risk_reward_ratio
        
        if side == 'long':
            stop_loss = entry_price * (1 - stop_distance)
            take_profit = entry_price * (1 + take_profit_distance)
        else:  # short
            stop_loss = entry_price * (1 + stop_distance)
            take_profit = entry_price * (1 - take_profit_distance)
        
        return stop_loss, take_profit
    
    def open_position(self, 
                     symbol: str,
                     side: str,
                     size: float,
                     entry_price: float,
                     confidence: float) -> bool:
        """
        Open a new position with risk management.
        
        Returns:
            True if position was opened successfully
        """
        # Check circuit breaker
        risk_metrics = self.calculate_risk_metrics()
        market_volatility = self.volatility_adjuster.calculate_volatility(symbol)
        
        if self.circuit_breaker.check_circuit_breaker(
            risk_metrics.current_drawdown,
            risk_metrics.total_pnl / self.initial_capital,
            market_volatility
        ):
            system_logger.warning(f"Cannot open position for {symbol}: Circuit breaker active")
            return False
        
        # Check if position already exists
        if symbol in self.positions:
            system_logger.warning(f"Position for {symbol} already exists")
            return False
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            symbol, entry_price, side, confidence
        )
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add position to portfolio
        self.positions[symbol] = position
        self.total_trades += 1
        
        # Update price history
        self.volatility_adjuster.add_price(symbol, entry_price)
        
        # Log trade
        position_value = size * entry_price
        audit_logger.info(
            f"Opened {side} position: {symbol} "
            f"size={size:.6f} price=${entry_price:.4f} "
            f"value=${position_value:.2f} "
            f"SL=${stop_loss:.4f} TP=${take_profit:.4f}"
        )
        
        return True
    
    def close_position(self, 
                      symbol: str, 
                      exit_price: float, 
                      reason: str = "manual") -> Optional[float]:
        """
        Close a position and return realized P&L.
        
        Returns:
            Realized P&L or None if position doesn't exist
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # Calculate realized P&L
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size
            
        # Update statistics
        is_winner = pnl > 0
        if is_winner:
            self.winning_trades += 1
            
        return_pct = pnl / (position.entry_price * position.size)
        self.kelly_calculator.update_statistics(is_winner, abs(return_pct))
        
        # Update capital
        self.current_capital += pnl
        
        # Move to closed positions
        position.unrealized_pnl = pnl
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update price history
        self.volatility_adjuster.add_price(symbol, exit_price)
        
        # Log trade close
        audit_logger.info(
            f"Closed {position.side} position: {symbol} "
            f"entry=${position.entry_price:.4f} exit=${exit_price:.4f} "
            f"pnl=${pnl:.2f} reason={reason}"
        )
        
        return pnl
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update all positions with current market data."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.update_pnl(current_price)
                
                # Check stop loss and take profit
                should_close = False
                close_reason = ""
                
                if position.side == 'long':
                    if current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price >= position.take_profit:
                        should_close = True
                        close_reason = "take_profit"
                else:  # short
                    if current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price <= position.take_profit:
                        should_close = True
                        close_reason = "take_profit"
                
                # Close position if needed
                if should_close:
                    self.close_position(symbol, current_price, close_reason)
                else:
                    # Update price history
                    self.volatility_adjuster.add_price(symbol, current_price)
    
    def get_current_exposure(self) -> float:
        """Calculate current total exposure across all positions."""
        total_exposure = 0.0
        for position in self.positions.values():
            position_value = abs(position.size * position.entry_price)
            total_exposure += position_value
        return total_exposure
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Current portfolio value
        current_portfolio_value = self.current_capital
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_portfolio_value = current_portfolio_value + unrealized_pnl
        
        # Update portfolio value history
        self.portfolio_value_history.append(total_portfolio_value)
        
        # Calculate drawdown
        self.peak_portfolio_value = max(self.peak_portfolio_value, total_portfolio_value)
        current_drawdown = (self.peak_portfolio_value - total_portfolio_value) / self.peak_portfolio_value
        
        # Calculate maximum drawdown
        if len(self.portfolio_value_history) > 1:
            portfolio_values = list(self.portfolio_value_history)
            peak = portfolio_values[0]
            max_dd = 0.0
            
            for value in portfolio_values:
                peak = max(peak, value)
                dd = (peak - value) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0.0
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            avg_return = np.mean(self.daily_returns)
            return_std = np.std(self.daily_returns)
            sharpe_ratio = (avg_return * 252) / (return_std * np.sqrt(252)) if return_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Portfolio exposure and heat
        total_exposure = self.get_current_exposure()
        portfolio_heat = total_exposure / self.current_capital
        
        # Win rate
        losing_trades = self.total_trades - self.winning_trades
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # VaR calculation (simplified)
        if len(self.portfolio_value_history) > 20:
            returns = []
            values = list(self.portfolio_value_history)
            for i in range(1, len(values)):
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
            
            var_95 = np.percentile(returns, 5) * total_portfolio_value
            var_99 = np.percentile(returns, 1) * total_portfolio_value
        else:
            var_95 = var_99 = 0.0
        
        return RiskMetrics(
            total_exposure=total_exposure,
            portfolio_heat=portfolio_heat,
            var_95=abs(var_95),
            var_99=abs(var_99),
            max_drawdown=max_dd,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_pnl=total_portfolio_value - self.initial_capital,
            winning_trades=self.winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        risk_metrics = self.calculate_risk_metrics()
        
        # Position details
        position_details = []
        for symbol, position in self.positions.items():
            position_details.append({
                'symbol': symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'unrealized_pnl': position.unrealized_pnl,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            })
        
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_value': self.current_capital + sum(pos.unrealized_pnl for pos in self.positions.values()),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            },
            'risk_metrics': {
                'total_exposure': risk_metrics.total_exposure,
                'portfolio_heat': risk_metrics.portfolio_heat,
                'max_heat_allowed': self.max_portfolio_heat,
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'max_drawdown': risk_metrics.max_drawdown,
                'current_drawdown': risk_metrics.current_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio
            },
            'trading_stats': {
                'total_trades': self.total_trades,
                'winning_trades': risk_metrics.winning_trades,
                'losing_trades': risk_metrics.losing_trades,
                'win_rate': risk_metrics.win_rate,
                'total_pnl': risk_metrics.total_pnl
            },
            'positions': position_details,
            'circuit_breaker': {
                'is_active': self.circuit_breaker.is_triggered,
                'reason': self.circuit_breaker.trigger_reason,
                'trigger_time': self.circuit_breaker.trigger_time.isoformat() if self.circuit_breaker.trigger_time else None
            }
        }


# Factory function
def create_risk_manager(initial_capital: float = 100000.0) -> DynamicRiskManager:
    """Create a Dynamic Risk Manager with specified initial capital."""
    return DynamicRiskManager(initial_capital)