# 🎓 Araştırmada Üretken Yapay Zekâ Kullanımı — Pratik Uygulama

> **BTK Akademi** | *Araştırmada Üretken Yapay Zekâ Kullanımı* eğitiminin pratik çıktıları

[![BTK Akademi](https://img.shields.io/badge/BTK%20Akademi-Katılım%20Sertifikalı-blue)](https://www.btkakademi.gov.tr)
[![Seviye](https://img.shields.io/badge/Seviye-Orta-yellow)]()
[![Araç](https://img.shields.io/badge/Araç-Claude%20AI-orange)](https://claude.ai)
[![Konu](https://img.shields.io/badge/Konu-Akıllı%20Yol%20Güvenliği-green)]()

---

## 📌 Bu Depo Hakkında

Bu depo, BTK Akademi'nin **"Araştırmada Üretken Yapay Zekâ Kullanımı"** başlıklı eğitiminde öğrenilen kavramları pratiğe dökmek amacıyla hazırlanmıştır.

Eğitimde öğretilen beceriler — akademik yazma, literatür tarama, veri analizi, tartışma ve sonuç yazımı, atıf düzenleme, çeviri ve intihal denetimi — **Claude AI** kullanılarak gerçek bir lisans araştırması üzerinde uygulanmıştır.

**Uygulama konusu:** *Yapay Zeka ve 5G Entegrasyonuyla Akıllı Yol Güvenliği: Gerçek Zamanlı Nesne Tespiti ve Sürücü Davranış Analizi Üzerine Betimsel Bir Araştırma*

---

## 🎯 Eğitim Bilgileri

| Alan | Detay |
|---|---|
| **Platform** | BTK Akademi |
| **Eğitim Adı** | Araştırmada Üretken Yapay Zekâ Kullanımı |
| **Seviye** | Orta |
| **Süre** | 5 saat 3 dakika |
| **Katılımcı** | ~7.385 |
| **Sertifika** | Katılım Sertifikalı |
| **Kullanılan Araç** | Claude AI (Anthropic) |

### Eğitimde Kapsanan Konular
- Üretken YZ'nın araştırma süreçlerinde etik kullanımı
- Akademik yazma ve yapılandırma
- Literatür tarama ve kaynak organizasyonu
- Veri analizi desteği
- Sonuç, tartışma ve özet yazımı
- Atıf düzenleme (APA 7)
- Çeviri ve dil kontrolü

---

## 📁 Dosya Yapısı

```
📦 repo
 ┣ 📄 README.md                        ← Bu dosya
 ┣ 📄 makale_nihai.md                  ← Türkçe nihai makale (tüm bölümler birleşik)
 ┣ 📄 makale_nihai_en.md               ← İngilizce çeviri
 ┣ 📄 ozet_yapilandirilmamis.md        ← Özet (yapılandırılmamış format)
 ┣ 📄 giris_bolumu.md                  ← 1. Giriş bölümü
 ┣ 📄 related_work.md                  ← 2. İlgili Çalışmalar (Related Work)
 ┣ 📄 arastirma_sorulari.md            ← 3. Araştırma Soruları
 ┣ 📄 yontem_arastirma_deseni.md       ← 4.1 Araştırma Deseni
 ┣ 📄 yontem_evren_orneklem.md         ← 4.2 Evren ve Örneklem
 ┣ 📄 tartisma_bolumu.md               ← 5. Tartışma
 ┣ 📄 sonuc_bolumu.md                  ← 6. Sonuç
 ┣ 📄 anahtar_kelimeler.md             ← Anahtar Kelimeler (TR + EN)
 ┣ 📄 baslik_onerileri.md              ← 10 Başlık Önerisi
 ┗ 📄 hakem_degerlendirmesi.md         ← Reviewer rolüyle makale değerlendirmesi
```

---

## 🔄 Üretim Süreci ve Kullanılan Promptlar

Her dosya için Claude AI'ye verilen prompt ve eğitimde karşılık geldiği beceri aşağıda listelenmiştir.

---

### 1. `related_work.md` — İlgili Çalışmalar (Literatür Tarama)

> **Eğitim Becerisi:** Literatür tarama ve kaynak organizasyonu

**Kullanılan Prompt:**
```
Ben bir lisans öğrencisiyim. "Yapay Zeka ile Akıllı Yol Güvenliği"ne yönelik bir araştırma
hazırlıyorum. Bu araştırmanın literatür bölümünde (related work) kullanmak için bu alanda
gerçekleştirilmiş çalışmaları özetlemeni istiyorum. Bu hususta son 5 yılda Türkiye, Avrupa,
Amerika ve Çin'de gerçekleştirilen çalışmaları araştırarak bulduğun çalışmaları: yazar, yıl,
yöntem ve sonuçlarını özetleyecek şekilde bir literatür bölümü hazırlamanı istiyorum.
Bölüm için 15-20 farklı güncel araştırmayı incelemeni istiyorum. Çalışmalara APA 7
formatına göre atıf yaparak bölüm sonunda kaynakça oluşturmanı istiyorum.
```

**Not:** Dosya ilk olarak `.docx` formatında, ardından GitHub'da paylaşılabilmesi için `.md` formatına dönüştürülmüştür.

---

### 2. `giris_bolumu.md` — Giriş Bölümü (Akademik Yazma)

> **Eğitim Becerisi:** Akademik yazma ve yapılandırma

**Kullanılan Prompt:**
```
Bu makale için giriş bölümü hazırlamanı istiyorum. Giriş bölümü hazırlarken şu hususları
dahil et:
"Konunun Tanıtımı / Problem Durumu ve Gereklilik / Bağlam Kurma / Literatürle
İlişkilendirme / Araştırmanın Amacı ve Sınırları / Özgün Katkının Vurgulanması"

Araştırmanın amacı; bilgisayarlı görü (YOLO vb.) ve makine öğrenmesi algoritmalarının,
yüksek hızlı veri aktarımı sağlayan teknolojilerle (5G vb.) entegre edilerek akıllı yol
güvenliğine sağlayacağı katkıları değerlendirmektir. Bu araştırma, gerçek zamanlı nesne
tespiti (yaya, araç, engel) ve otonom uyarı mekanizmaları ile sınırlandırılmıştır.
Araştırmanın, literatürdeki mevcut çalışmaları sentezleyerek sistemlerdeki gecikme/doğruluk
eksikliklerini tespit etmesi, otonom sistemler ve akıllı yol altyapıları için geliştirilecek
yenilikçi projelere teorik bir zemin hazırlaması yönünden özgün katkıları bulunmaktadır.
```

---

### 3. `arastirma_sorulari.md` — Araştırma Soruları

> **Eğitim Becerisi:** Akademik yapılandırma ve problem formülasyonu

**Kullanılan Prompt:**
```
Şimdi bu araştırma için bana literatürdeki sınırlılıkları ve benim amaçlarımı dikkate
alarak 5 farklı araştırma sorusu önermeni istiyorum.
```

**İkinci Prompt (dosyalaştırma):**
```
Bu araştırma sorularının yer aldığı bölümü .md olarak oluşturmanı istiyorum.
```

---

### 4. `yontem_arastirma_deseni.md` — Araştırma Deseni

> **Eğitim Becerisi:** Akademik yazma — yöntem bölümü

**Kullanılan Prompt:**
```
Bu araştırma için 'Yöntem' (Methodology) bölümü için bir alt başlık metni hazırlamanı
istiyorum. Araştırmamda Nicel Araştırma Yöntemi ve alt desen olarak Betimsel Desen
(Descriptive Design) kullanıyorum. Lütfen metni hazırlarken şu hususları dahil et:

• Gerekçelendirme: Betimsel desenin bu çalışma için neden seçildiğini akıllı ulaşım
  sistemleri bağlamında açıkla
• Veri Tipi: Sayısal verilerin araştırmadaki rolünü vurgula
• Dil ve Üslup: Akademik, nesnel ve 3. tekil şahıs bir dil kullan
• Araştırmanın doğası gereği değişkenler üzerinde bir oynama yapılmadığını, mevcut
  durumun sayısal bir "fotoğrafının çekildiğini" akademik bir dille ifade et.
```

---

### 5. `yontem_evren_orneklem.md` — Evren ve Örneklem

> **Eğitim Becerisi:** Akademik yazma — yöntem bölümü

**Kullanılan Prompt:**
```
Araştırmanın "Evren ve Örneklem (Veri Seti ve Sınıflandırma)" alt bölümü için şu bilgiler
ışığında akademik bir metin oluştur. Çalışmanın örneklemi olarak, modelin (örn. YOLO vb.)
eğitimi ve testi için Roboflow (veya benzeri platformlar) üzerinden etiketlenmiş özel bir
görüntü veri seti kullanılmıştır. Bu örneklem büyüklüğü ve çeşitliliği, literatürdeki nesne
tespiti (object detection) ve bilgisayarlı görü çalışmalarından çıkarım yapılarak
belirlenmiştir. Veri setinde dengeyi sağlamak amacıyla farklı hava koşulları, günün farklı
saatleri (gece/gündüz) ve yaya/araç çeşitliliği göz önüne alınmış; veriler eğitim (train),
doğrulama (validation) ve test kümelerine spesifik oranlarda bölünerek araştırmaya dahil
edilmiştir. Modelin performansına etki edebilecek diğer çevresel faktörler veri setindeki
çeşitlilik üzerinden analiz edilmiştir.
```

---

### 6. `tartisma_bolumu.md` — Tartışma Bölümü

> **Eğitim Becerisi:** Sonuç ve tartışma yazımı

**Kullanılan Prompt:**
```
Ekte sunduğum (veya bulgularını paylaştığım) "Yapay Zeka ile Akıllı Yol Güvenliği" konulu
makalemin "Tartışma" (Discussion) bölümünü oluşturmanı istiyorum. Akademik bir dille ve
APA 7.0 formatını kullanarak şu hususları içeren bir metin hazırla:

• Araştırmadan elde edilen nicel bulguları (bilgisayarlı görü, özellikle YOLOv8 gibi
  modellerin performans metriklerini ve 5G'nin sağladığı düşük gecikme avantajlarını)
  literatürdeki geçmiş çalışmalarla karşılaştırarak yorumla.
• Mevcut sistemlerin eksiklikleri ile bizim bulgularımızın nasıl örtüştüğünü veya
  ayrıştığını analiz et.
• Modelin gerçek zamanlı nesne tespitindeki doğruluk ve hız potansiyelinin akıllı şehir
  altyapılarına etkisini tartış.
• Son olarak, bu çalışmanın literatüre ve gelecekteki otonom sürüş sistemlerine sunduğu
  özgün katkıları net bir şekilde vurgula.
```

---

### 7. `sonuc_bolumu.md` — Sonuç Bölümü

> **Eğitim Becerisi:** Akademik yazma — sonuç yazımı

**Kullanılan Prompt:**
```
Bu makalenin sonuç bölümünü 1-2 paragraf olacak şekilde akademik bir formatta
oluşturmanı istiyorum.
```

---

### 8. `ozet_yapilandirilmamis.md` — Özet

> **Eğitim Becerisi:** Akademik özet yazımı

**İlk Prompt (yapılandırılmış özet):**
```
Bu çalışma için arkaplan, amaç, yöntem, bulgular ve sonuç bölümlerini içeren
yapılandırılmış bir özet oluşturmanı istiyorum. Özet 250 kelime ile sınırlandırılsın.
```

**İkinci Prompt (yapılandırılmamış formata dönüştürme):**
```
Bu özeti yapılandırılmamış bir özete dönüştür.
```

---

### 9. `anahtar_kelimeler.md` — Anahtar Kelimeler

> **Eğitim Becerisi:** Akademik yazma — anahtar kelime seçimi

**Kullanılan Prompt:**
```
Bu araştırma için 10 tane anahtar kelime öner.
```

**İkinci Prompt (dosyalaştırma):**
```
.md dosyasına yaz, indireceğim.
```

---

### 10. `baslik_onerileri.md` — Başlık Önerileri

> **Eğitim Becerisi:** Akademik yazma — başlık formülasyonu

**Kullanılan Prompt:**
```
"Yapay Zeka ile Akıllı Yol Güvenliği" konulu akademik araştırmam için
10 tane başlık önerisinde bulun.
```

**İkinci Prompt (dosyalaştırma):**
```
Hepsini .md dosyasına yaz, indireceğim.
```

---

### 11. `makale_nihai.md` — Nihai Makale (Türkçe)

> **Eğitim Becerisi:** Tüm bölümlerin bütünleştirilmesi

**Kullanılan Prompt:**
```
Tüm dosyaları birleştirerek nihai makalemi oluşturmanı istiyorum.
```

---

### 12. `makale_nihai_en.md` — Nihai Makale (İngilizce Çeviri)

> **Eğitim Becerisi:** Çeviri

**Kullanılan Prompt:**
```
Nihai makaleyi İngilizce'ye çevirmeni istiyorum.
```

---

### 13. `hakem_degerlendirmesi.md` — Hakem Değerlendirmesi

> **Eğitim Becerisi:** Eleştirel değerlendirme ve geri bildirim yorumlama

**Kullanılan Prompt:**
```
Türkçe nihai makalenin taslağını bir reviewer rolüyle değerlendirmeni istiyorum.
Makalenin güçlü ve zayıf yönlerini listelemeni, ayrıca sınırlılıklarını nasıl
güçlendirebileceğimi de önerilerle belirtmeni istiyorum.
```

---

## 🗺️ Eğitim Becerisi → Dosya Eşleştirmesi

| Eğitim Becerisi | Üretilen Dosya |
|---|---|
| Literatür tarama ve kaynak organizasyonu | `related_work.md` |
| Akademik yazma — giriş | `giris_bolumu.md` |
| Problem formülasyonu | `arastirma_sorulari.md` |
| Akademik yazma — yöntem | `yontem_arastirma_deseni.md`, `yontem_evren_orneklem.md` |
| Sonuç ve tartışma yazımı | `tartisma_bolumu.md`, `sonuc_bolumu.md` |
| Özet yazımı | `ozet_yapilandirilmamis.md` |
| Atıf düzenleme (APA 7) | Tüm bölümler — kaynakça entegre |
| Anahtar kelime seçimi | `anahtar_kelimeler.md` |
| Başlık formülasyonu | `baslik_onerileri.md` |
| Çeviri | `makale_nihai_en.md` |
| Eleştirel değerlendirme | `hakem_degerlendirmesi.md` |
| Bütünleştirme | `makale_nihai.md` |

---

## ⚙️ Kullanılan Araçlar

| Araç | Kullanım Amacı |
|---|---|
| **Claude AI** (Anthropic) | Tüm akademik içerik üretimi |
| **Roboflow** | Veri seti etiketleme (araştırmanın kendisi) |
| **YOLOv11 / RT-DETR** | Model mimarileri (araştırmanın kendisi) |

---

## ⚠️ Önemli Notlar

**Akademik dürüstlük hakkında:**
Bu depo bir eğitim pratiği çalışmasıdır. Üretken YZ ile oluşturulan akademik içerikler gerçek bir araştırma sürecinin parçasıdır ve YZ kullanımı şeffaf biçimde belgelenmiştir. Gerçek bir yayına dönüştürülmeden önce hakem değerlendirmesinde tespit edilen eksiklikler (bkz. `hakem_degerlendirmesi.md`) tamamlanmalıdır.

**Veri kaynakları hakkında:**
`related_work.md` içindeki tüm çalışmalar gerçek yayınlara dayanmaktadır. Model performans değerleri (`makale_nihai.md`) bu araştırma kapsamında gerçekleştirilen model eğitim deneylerinden elde edilmiştir.

---

## 📜 Lisans

Bu depo eğitim amaçlı hazırlanmıştır. İçerikler serbestçe referans alınabilir; kaynak gösterilmesi rica olunur.

---

*BTK Akademi — Araştırmada Üretken Yapay Zekâ Kullanımı eğitiminin pratik uygulaması olarak hazırlanmıştır.*
