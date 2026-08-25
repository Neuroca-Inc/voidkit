#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

typedef unsigned __int128 u128;
static inline u128 mk128(uint64_t hi,uint64_t lo){return ((u128)hi<<64)|(u128)lo;}
static inline void p128(FILE*f,u128 x){char b[64];int n=0;if(!x){fputc('0',f);return;}while(x){b[n++]='0'+x%10;x/=10;}while(n)fputc(b[--n],f);}
static inline uint64_t floor_lo(u128 x){return (uint64_t)(x>>96);}
static inline uint64_t floor_strict_hi(u128 x){return (uint64_t)((x-1)>>96);}

static inline int threshold(uint64_t j,int terminal,uint64_t*out){
 if(j<=2){*out=1;return 1;}
 const u128 alo=mk128(0x170bf5efcULL,0xd2c35844f742c488ULL);
 const u128 ahi=mk128(0x170bf5efcULL,0xd2c35844f742c489ULL);
 const u128 blo=mk128(0x2c1a46a0ULL,0x63eff5ba65eb3d4fULL);
 const u128 bhi=mk128(0x2c1a46a0ULL,0x63eff5ba65eb3d50ULL);
 u128 lo=alo*(u128)j+blo, hi=ahi*(u128)j+bhi;
 if(terminal){
  uint64_t a=floor_lo(lo)+1,b=floor_strict_hi(hi)+1;
  if(a!=b){return 0;}
  *out=a;
  return 1;
 }
 uint64_t a=floor_lo(lo),b=floor_strict_hi(hi);
 if(a!=b){return 0;}
 *out=a;
 return 1;
}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(int argc,char**argv){
 uint64_t limit=argc>1?strtoull(argv[1],0,10):1000000000ULL;
 const char*outpath=argc>2?argv[2]:"L_CHECKPOINTS.csv";
 FILE*out=fopen(outpath,"w");if(!out){perror("fopen");return 2;}
 fprintf(out,"closed_A,L_tick,B_total,Q_total,layer_B,layer_Q,layer_points,carry,completed_layers,total_points,within_edges,directed_cross_placements,total_relation_entries,chart_point_entries,transfer_entries_each_direction,word_hash_a,word_hash_b\n");
 uint64_t tick=0,B=0,Q=0,L=0,A=0,k=0,j=1,N=6,active=2,completed_points=0;
 uint64_t Bstart=0,Qstart=0,prevT=0,ambiguities=0;
 u128 sumsq_completed=0;
 u128 completed_mutation_residual=0;
 uint64_t h1=1469598103934665603ULL,h2=0x9e3779b97f4a7c15ULL;
 char first64[65]={0};
 double t0=now_s();
 while(tick<limit){
  int terminal=(k==N-1);uint64_t target=0;char p=0;int is_l=0;
  uint64_t completed_points_before=completed_points;u128 sumsq_completed_before=sumsq_completed;
  uint64_t closedA=0,layerB=0,layerQ=0,layerPoints=0;int64_t carry=0;
  if(!threshold(j,terminal,&target)){ambiguities++;fprintf(stderr,"threshold ambiguity at j=%"PRIu64"\n",j);return 3;}
  if(B<target){p='B';B++;active++;}
  else if(!terminal){p='Q';Q++;k++;j++;}
  else {
   p='L';is_l=1;L++;closedA=A;layerB=B-Bstart;layerQ=Q-Qstart;layerPoints=active;
   carry=(closedA==0)?0:(int64_t)B-2*(int64_t)prevT;prevT=B;
   completed_points+=active;sumsq_completed+=(u128)active*active;
   A++;k=0;N=6ULL<<A;j=1+6*((1ULL<<A)-1);Bstart=B;Qstart=Q;active=2;
  }
  if(!is_l){
   completed_mutation_residual += (u128)(completed_points_before != completed_points);
   completed_mutation_residual += (u128)(sumsq_completed_before != sumsq_completed);
  }
  if(tick<64)first64[tick]=p;
  h1^=(unsigned char)p;h1*=1099511628211ULL;
  h2^=(uint64_t)(unsigned char)p+0x9e3779b97f4a7c15ULL+(h2<<6)+(h2>>2);h2=(h2<<17)|(h2>>47);
  tick++;
  if(is_l){
   uint64_t S=completed_points+active;u128 q2=sumsq_completed+(u128)active*active;
   u128 within=(q2-S)/2,cross=(u128)S*S-q2,total=within+cross;
   fprintf(out,"%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",",closedA,tick,B,Q,layerB,layerQ,layerPoints);
   if(closedA==0)fprintf(out,",");else fprintf(out,"%"PRId64",",carry);
   fprintf(out,"%"PRIu64",%"PRIu64",",L,S);p128(out,within);fputc(',',out);p128(out,cross);fputc(',',out);p128(out,total);
   fprintf(out,",%"PRIu64",%"PRIu64",%016"PRIx64",%016"PRIx64"\n",S,S,h1,h2);
  }
 }
 double elapsed=now_s()-t0;fclose(out);
 uint64_t S=completed_points+active;u128 q2=sumsq_completed+(u128)active*active;
 u128 within=(q2-S)/2,cross=(u128)S*S-q2,total=within+cross;
 printf("{\n\"ticks\":%"PRIu64",\n\"B\":%"PRIu64",\n\"Q\":%"PRIu64",\n\"L\":%"PRIu64",\n\"A\":%"PRIu64",\n\"k\":%"PRIu64",\n\"j\":%"PRIu64",\n\"active_points\":%"PRIu64",\n\"completed_points\":%"PRIu64",\n\"total_points\":%"PRIu64",\n\"within_edges\":\"",tick,B,Q,L,A,k,j,active,completed_points,S);
 p128(stdout,within);printf("\",\n\"directed_cross_placements\":\"");p128(stdout,cross);printf("\",\n\"total_relation_entries\":\"");p128(stdout,total);
 printf("\",\n\"chart_point_entries\":%"PRIu64",\n\"transfer_entries_each_direction\":%"PRIu64",\n\"completed_layer_mutation_residual\":\"",S,S);p128(stdout,completed_mutation_residual);
 printf("\",\n\"threshold_ambiguities\":%"PRIu64",\n\"first64\":\"%s\",\n\"primitive64\":\"%c\",\n\"word_hash_a\":\"%016"PRIx64"\",\n\"word_hash_b\":\"%016"PRIx64"\"\n}\n",ambiguities,first64,first64[63],h1,h2);
 fprintf(stderr,"{\"elapsed_seconds\":%.9f,\"ticks_per_second\":%.3f}\n",elapsed,limit/elapsed);
 return 0;
}
