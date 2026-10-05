"""
Gestão de risco: sizing de posição, stop loss, take profit e validação de R:R.
Toda posição passa obrigatoriamente por estas funções antes de entrar.
"""


def calc_position_size(balance_usdt: float, risk_pct: float,
                       entry: float, stop_loss: float) -> float:
    """
    Calcula o tamanho da posição em USDT baseado no risco percentual.

    Exemplo: balance=1000, risk_pct=1.0, entry=100, stop=95
    → sl_distance = 5 (5%)
    → risk_usdt = 10 (1% de 1000)
    → position_size = 10 / 0.05 = 200 USDT (com alavancagem necessária)

    Retorna o valor notional em USDT da posição.
    """
    sl_distance_pct = abs(entry - stop_loss) / entry
    if sl_distance_pct == 0:
        return 0.0

    risk_usdt = balance_usdt * (risk_pct / 100)
    position_size = risk_usdt / sl_distance_pct
    return round(position_size, 2)


def calc_stop_loss(entry: float, atr: float, side: str,
                   atr_mult: float = 2.0) -> float:
    """
    Stop loss baseado em ATR.
    side: 'long' ou 'short'
    """
    if side == "long":
        return round(entry - atr * atr_mult, 8)
    else:
        return round(entry + atr * atr_mult, 8)


def calc_take_profit(entry: float, stop_loss: float, side: str,
                     rr: float = 3.0) -> float:
    """
    Take profit baseado no R:R desejado.
    rr=3.0 → alvo é 3× a distância do stop.
    """
    risk = abs(entry - stop_loss)
    if side == "long":
        return round(entry + risk * rr, 8)
    else:
        return round(entry - risk * rr, 8)


def calc_rr(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calcula o R:R real de uma configuração de trade."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def calc_required_leverage(balance_usdt: float, position_size_usdt: float) -> int:
    """
    Calcula a alavancagem necessária para a posição dado o saldo disponível.
    Limita a 20x (máximo permitido no bot).
    """
    if balance_usdt == 0:
        return 1
    leverage = position_size_usdt / balance_usdt
    return max(1, min(20, int(leverage) + 1))


def validate_trade(entry: float, stop_loss: float, take_profit: float,
                   rr_minimo: float, side: str) -> tuple[bool, str]:
    """
    Valida se um trade pode ser aberto.
    Retorna (True, "") se OK, ou (False, motivo) se rejeitado.
    """
    # Stop na direção errada
    if side == "long" and stop_loss >= entry:
        return False, "Stop loss acima do preço de entrada para LONG"
    if side == "short" and stop_loss <= entry:
        return False, "Stop loss abaixo do preço de entrada para SHORT"

    # Take profit na direção errada
    if side == "long" and take_profit <= entry:
        return False, "Take profit abaixo do preço de entrada para LONG"
    if side == "short" and take_profit >= entry:
        return False, "Take profit acima do preço de entrada para SHORT"

    # R:R insuficiente
    rr = calc_rr(entry, stop_loss, take_profit)
    if rr < rr_minimo:
        return False, f"R:R insuficiente: {rr:.2f} (mínimo {rr_minimo:.1f})"

    return True, ""
