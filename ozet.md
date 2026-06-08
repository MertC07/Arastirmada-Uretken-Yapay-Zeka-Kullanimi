# Özet

**Arkaplan:** Karayolu trafik kazaları küresel ölçekte önde gelen ölüm nedenlerinden biri olmayı sürdürmekte; Dünya Sağlık Örgütü 2021 yılında 1,19 milyon trafik kaynaklı ölüm bildirmektedir. Türkiye'de ise 2023 yılında 6.548 kişi hayatını kaybetmiş, kazaların %90'ından fazlasında insan hatası belirleyici etken olmuştur. Bilgisayarlı görü algoritmalarının ve 5G altyapısının olgunlaşması, proaktif ve gerçek zamanlı yol güvenliği sistemleri için yeni bir pencere açmaktadır.

**Amaç:** Bu araştırma; YOLO serisi ve RT-DETR gibi derin öğrenme tabanlı nesne tespit mimarilerinin 5G destekli kenar hesaplama altyapısıyla entegrasyonunun akıllı yol güvenliğine sağlayabileceği katkıları değerlendirmeyi amaçlamaktadır. Araştırma; gerçek zamanlı nesne tespiti (araç, yaya, engel) ve otonom uyarı mekanizmalarıyla sınırlandırılmıştır.

**Yöntem:** Nicel araştırma paradigması çerçevesinde betimsel desen benimsenmiştir. Roboflow platformu aracılığıyla derlenen etiketlenmiş görüntü veri seti %70 eğitim, %20 doğrulama ve %10 test kümelerine ayrılmıştır. YOLOv11l, YOLOv11n+αSiLU, RT-DETR-L ve YOLOv8n mimarileri; plaka tespiti, sürücü davranış analizi ve emniyet kemeri denetimi görevlerinde mAP, Precision ve Recall metrikleriyle karşılaştırmalı olarak değerlendirilmiştir.

**Bulgular:** Plaka tespitinde YOLOv11l (mAP@50: 0,986), telefon kullanımı tespitinde RT-DETR-L (Precision: 1,000) ve yorgunluk göstergelerinin tanınmasında YOLOv11n+αSiLU (esneme mAP@50: 0,995) en yüksek başarımı sergilemiştir. Emniyet kemeri tespiti, tüm modellerde en kritik performans açığı olarak belirlenmiştir.

**Sonuç:** Hiçbir tek mimarinin tüm güvenlik görevlerinde üstünlük sağlayamadığı ampirik olarak ortaya konmuş; görev bazlı modüler sistem tasarımının gerekliliği desteklenmiştir. Bulgular, otonom uyarı mekanizmaları ve akıllı şehir altyapıları için geliştirilecek sistemlere teorik zemin sunmaktadır.

**Anahtar Kelimeler:** Akıllı yol güvenliği, bilgisayarlı görü, nesne tespiti, YOLOv11, RT-DETR, 5G, sürücü davranış analizi, derin öğrenme
