

def get_diagnosis(yes_count):
    # تحديد التوصية بناءً على عدد الإجابات بـ "نعم"
    if yes_count == 10:
        return "نعم: 10 لا: 0 - الطفل سليم."
    elif yes_count == 9:
        return "نعم: 9 لا: 1 - تصرفات طبيعية."
    elif yes_count == 8:
        return "نعم: 8 لا: 2 - تصرفات طبيعية لا داعي للقلق."
    elif yes_count == 7:
        return "نعم: 7 لا: 3 - حالة توحد خفيفة وينصح بدمج الطفل مع أطفال آخرين."
    elif yes_count == 6:
        return "نعم: 6 لا: 4 - حالة توحد خفيفة ويفضل مراقبة الطفل."
    elif yes_count == 5:
        return "نعم: 5 لا: 5 - حالة توحد متوسطة تستدعي مراقبة الطفل."
    elif yes_count in [3, 4]:
        return "نعم: {} لا: {} - حالة توحد متوسطة وعند تكرار التصرفات ينصح بزيارة المختص.".format(yes_count, 10 - yes_count)
    elif yes_count in [1, 2]:
        return "نعم: {} لا: {} - حالة توحد متقدمة وينصح بزيارة المختص.".format(yes_count, 10 - yes_count)
    else:
        return "نعم: 0 لا: 10 - حالة توحد متقدمة وتحتاج إلى استشارة طبية عاجلة."

    pass
