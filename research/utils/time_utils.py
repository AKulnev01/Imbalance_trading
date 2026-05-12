# utils/time_utils.py
def format_timestamp(ts):
    """
    Форматирует pandas.Timestamp или datetime в строку вида ДД/ММ/ГГГГ ЧЧ:ММ.
    """
    return ts.strftime("%d/%m/%Y %H:%M")