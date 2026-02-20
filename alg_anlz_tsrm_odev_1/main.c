#include <stdio.h>              // Umut Tüter 1240505901
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define K_SERVERS 5             // Sunucu sayýsý
#define NUM_REQUESTS 10000      // Simüle edilecek toplam istek
#define TAU 0.5                 // Softmax Sýcaklýk (Temperature) parametresi
#define ALPHA 0.1               // Öðrenme katsayýsý (Non-stationary ortamlar için sabit adým boyu)

// Sunucu Yapýsý (Multi-Armed Bandit Kolu)
typedef struct {
    int id;
    double true_mean_latency; // Gerçek gecikme ortalamasý (Zamanla deðiþir)
    double estimated_reward;  // Q-deðeri (Ajanýn tahmini ödülü)
    int request_count;        // Sunucuya giden istek sayýsý
} Server;

// [Yardýmcý Fonksiyon] 0 ile 1 arasýnda rastgele sayý üretir
double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

// [Yardýmcý Fonksiyon] Box-Muller dönüþümü ile Gaussian (Normal) gürültü üretir
double rand_gaussian(double mean, double stddev) {
    double u1 = rand_double();
    double u2 = rand_double();
    double z0;
    
    // Logaritma içine 0 gelmesini engellemek için küçük bir kontrol
    if (u1 <= 1e-7) u1 = 1e-7; 
    z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

// [Ana Algoritma] Softmax Action Selection (Nümerik Stabilite ile)
int select_server_softmax(Server servers[], int k, double tau) {
    int i; // C89 standardý için döngü deðiþkeni baþta tanýmlandý
    double max_reward;
    double sum_exp = 0.0;
    double probabilities[K_SERVERS];
    double r, cumulative;

    // 1. NÜMERÝK STABÝLÝTE: En yüksek tahmini ödülü (Q deðerini) bul
    max_reward = servers[0].estimated_reward;
    for (i = 1; i < k; i++) {
        if (servers[i].estimated_reward > max_reward) {
            max_reward = servers[i].estimated_reward;
        }
    }

    // 2. Olasýlýklarý hesapla (Softmax Trick uygulayarak)
    for (i = 0; i < k; i++) {
        // max_reward çýkarýlarak overflow (taþma) engellenir
        probabilities[i] = exp((servers[i].estimated_reward - max_reward) / tau);
        sum_exp += probabilities[i];
    }

    // 3. Deðerleri normalize et (Toplamý 1 olacak þekilde oranla)
    for (i = 0; i < k; i++) {
        probabilities[i] /= sum_exp;
    }

    // 4. Rulet Tekerleði (Roulette Wheel) seçimi ile sunucuyu belirle
    r = rand_double();
    cumulative = 0.0;
    for (i = 0; i < k; i++) {
        cumulative += probabilities[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return k - 1; // Hata durumunda son sunucuyu dön
}

int main() {
    Server servers[K_SERVERS];
    int i, step; // Döngü deðiþkenleri baþta tanýmlandý
    int chosen_server;
    double observed_latency, reward;
    double total_latency = 0.0;
    clock_t start_time, end_time;
    double cpu_time_used;

    srand(time(NULL));

    // 1. Sunucularýn Baþlangýç Durumlarý (Initialization)
    for (i = 0; i < K_SERVERS; i++) {
        servers[i].id = i;
        servers[i].true_mean_latency = 50.0 + (rand() % 50); // 50ms - 100ms arasý rastgele baþla
        servers[i].estimated_reward = 0.0; // Henüz hiçbir þey bilmiyoruz
        servers[i].request_count = 0;
    }

    // Çalýþma Zamaný Analizi için saati baþlat
    start_time = clock();

    // 2. Simülasyon Döngüsü (Gelen Ýstekler)
    for (step = 0; step < NUM_REQUESTS; step++) {
        
        // A) Softmax ile en uygun sunucuyu seç
        chosen_server = select_server_softmax(servers, K_SERVERS, TAU);
        
        // B) Seçilen sunucudan yanýt al (Gürültülü / Noisy ortam simülasyonu)
        observed_latency = rand_gaussian(servers[chosen_server].true_mean_latency, 5.0);
        if (observed_latency < 1.0) observed_latency = 1.0; // Gecikme negatif veya 0 olamaz

        total_latency += observed_latency;
        servers[chosen_server].request_count++;

        // C) Ödülü hesapla (Gecikme ne kadar düþükse, ödül o kadar yüksektir)
        reward = -observed_latency; 

        // D) Q-Deðerini Güncelle (Non-Stationary ortamlar için Sabit Adým Boyu)
        servers[chosen_server].estimated_reward += ALPHA * (reward - servers[chosen_server].estimated_reward);

        // E) Non-Stationary Simülasyonu: Ortam sürekli deðiþiyor
        for (i = 0; i < K_SERVERS; i++) {
            servers[i].true_mean_latency += rand_gaussian(0.0, 0.5); 
        }
    }

    // Çalýþma Zamaný Analizi için saati durdur
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    // 3. Sonuçlarý Ekrana Yazdýr
    printf("--- Simulasyon Tamamlandi ---\n");
    printf("Toplam Istek Sayisi: %d\n", NUM_REQUESTS);
    printf("Toplam Gecikme (Latency): %.2f ms\n", total_latency);
    printf("Ortalama Gecikme: %.2f ms\n", total_latency / NUM_REQUESTS);
    printf("Algoritma Calisma Zamani: %f saniye\n\n", cpu_time_used);

    printf("--- Sunucu Istatistikleri ---\n");
    for (i = 0; i < K_SERVERS; i++) {
        printf("Sunucu %d | Istek Sayisi: %d | Son Gercek Ortalama Gecikme: %.2f ms | Tahmini Odul (Q): %.2f\n",
               servers[i].id, 
               servers[i].request_count, 
               servers[i].true_mean_latency, 
               servers[i].estimated_reward);
    }

    return 0;
}
