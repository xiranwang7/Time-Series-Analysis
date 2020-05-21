Time Series Analysis on U.S. Suicide Rate
================
Xiran Wang
04/22/2019











I.Background and Project Introduction
-------------------------------------

Every year, a million adults report attempting a suicide , according to the U.S. Centers for Disease Control and Prevention (CDC). Globally, 800,000 people die due to suicide every year, according to the World Health Organization (WHO). Many people look at suicide as a mystery, because at an individual level, it is hard to see any sign that leads towards suicide. At a society level, suicides usually involve mental illness caused by various reasons. As a result, preventing suicides seems to be more difficult. In addition, prevention suicide involve study of many subjects, such as psychology, ethology, sociology, and statistics, etc.

In this project, from the statistical perspective, as it will address changes of the U.S annual suicide rate (1920-2015), and the one-step ahead prediction values, and the 10-steps ahead prediction values, etc. The time-series model is built based on analyzing the annual suicide rat data (per 100,000) from 1920 to 2015. Based on the reasonable model, it predicts suicidal rates from 1940 to 2016 by using the data from prior years of the predicting year. In addition, it will also provide the 10-steps ahead predictions for suicide rates from 2016 to 2020. Furthermore, the 95% prediction interval for every predicted value will also be discussed in this article. More details will be addressed later. The main achievements of this project are summarized below,

-   Inferred 1940-2016 suicide rate in U.S. using time series ARIMA model.

-   Maintained the model’s accuracy prediction for extremely unusual events, such as The Great Depression.

-   Minimized average prediction error down to 0.72, secured 100% of predictions within 95% confidence interval.

II. Model Specification
-----------------------

### Step 1. Observation on the Raw Data Set

The plot of the raw data shows that there is significant spike, and that the raw data set looks quite nonstationary. It could cause inaccuracy to furfure model fitting. As a result, it is reasonable to consider calculating a difference on the raw data set. After calculating difference, the differenced data improves much better. The data is closer to stationary except some extreme values. Similarly, log- transformation is another common method to deal with raw data. However, there is not much change when it is used on raw data set. When taking the difference on logged data, the differenced logged data improvement looks the as good as the deferenced original data, so it is good to use a simpler one to fit the models.

``` r
# Data process
suicide.ts <- ts(suicide[,2], start=c(1920), frequency=1)

sr <- window(suicide.ts,start=1920,end=2015)

plot(sr,type='o', main='U.S Suicide Rate from 1920 to 2015', 
     cex.main=c(0.8), cex.axis=c(0.8))
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
lg    <- log(sr)
df1   <- diff(sr)
dflog <- diff(lg)

plot(df1,type='o', main='Differenced U.S Suicide Rate Data', 
     cex.main=c(0.8), cex.axis=c(0.8))
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r
plot(dflog,type='o', main='Differenced log U.S Suicide Rate Data', 
     cex.main=c(0.8), cex.axis=c(0.8))
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-2-3.png)

Based on observing the ACF plot, there is a very significant lag in lag-1. Indeed, the lag in lag-11 is also a little bit significant, but at this moment, to avoid having too many minor coefficients in the potential model, the lag-11 needs to be ignored temporarily. If the model fitting in the later steps doesn’t perform well, then we will go here and take care of the lag-11.

``` r
par(mfrow=c(1,2))

acf(df1)
pacf(df1)
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-3-1.png)

### Step 2. Comparision on candidate models

Based on observing the PACF, there is only one significant lag in lag-1. Combine the evidences from both ACF and PACF, start fitting the model by using ARIMA(1,1,1), ARIMA(0,1,1) and ARIMA(1,1,0). In the fact, the ARIMA(0,1,1) will be the best candidate model for the data set since it has the smallest both AIC and BIC values. The conparsion is in the following.

``` r
par.est <- function(p,q){
  
    model <- arima(sr, order=c(p,1,q),method='ML') # ARIMA(1,1,1) using maximum likelihood
    coeff <- model$coef
    aic   <- AIC(model)
    bic   <- BIC(model)
    out   <- list(coeff,aic,bic)
    return(out)
}


coef  <- rbind(cbind(par.est(1,0)[[1]],NA), cbind(NA,par.est(0,1)[[1]]),  par.est(1,1)[[1]] ) 
AICs  <- rbind(par.est(1,0)[2],par.est(0,1)[2],par.est(1,1)[2] )
BICs  <- rbind( par.est(1,0)[3],par.est(0,1)[3],par.est(1,1)[3])
table <- cbind(coef,AICs,BICs)

colnames(table) <- c("ar1 coef","ma1 coef","AIC","BIC")
rownames(table) <- c("ARIMA(1,1,0)","ARIMA(0,1,1)","ARIMA(1,1,1)")

kable(table)%>%
  kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
ar1 coef
</th>
<th style="text-align:left;">
ma1 coef
</th>
<th style="text-align:left;">
AIC
</th>
<th style="text-align:left;">
BIC
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
ARIMA(1,1,0)
</td>
<td style="text-align:left;">
0.29086267580524
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:left;">
172.445240269251
</td>
<td style="text-align:left;">
177.552994052452
</td>
</tr>
<tr>
<td style="text-align:left;">
ARIMA(0,1,1)
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:left;">
0.430136803674024
</td>
<td style="text-align:left;">
169.162538614682
</td>
<td style="text-align:left;">
174.270292397883
</td>
</tr>
<tr>
<td style="text-align:left;">
ARIMA(1,1,1)
</td>
<td style="text-align:left;">
-0.20223969769846
</td>
<td style="text-align:left;">
0.591256222786664
</td>
<td style="text-align:left;">
170.431672880104
</td>
<td style="text-align:left;">
178.093303554906
</td>
</tr>
</tbody>
</table>
III. Fitting and diagnostics
----------------------------

``` r
arma.d <- arima(sr, order=c(0,1,1),method='ML') # ARIMA(1,1,1) using maximum likelihood
```

After setting up the ARIMA(0,1,1) model, a model diagnostics was performed. Based on the plot of residuals, the most residuals are around zero except some possible outliers where the variance it is larger in the beginning. It is close to the white noise. The Q-Q plot reveals evidence of normality, but it indicates the possible outliers on both the tail and the head. Based on the ACF of residuals, the autocorrelations can be assumed to be minimal since all of them stay well within the boundaries. The Box-Ljung test has p-value = 0.248, which shows a strong evidence against the fact that the errors are not independent, thus ARIMA(0,1,1) model is supported by diagnostics.

``` r
arma.d <- arima(sr, order=c(0,1,1),method='ML') # ARIMA(1,1,1) using maximum likelihood
Box.test(arma.d$resid,type="Ljung-Box")
```

    ## 
    ##  Box-Ljung test
    ## 
    ## data:  arma.d$resid
    ## X-squared = 1.3346, df = 1, p-value = 0.248

``` r
par(mfrow=c(1,2))
qqnorm(residuals(arma.d),cex.main=c(0.7))

plot(residuals(arma.d),type='p',main='Residuals for ARIMA(0,1,1)',cex.main=c(0.7))
abline(h=0)
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
par(mfrow=c(1,1))

tsdiag(arma.d,cex.main=c(0.7))
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-6-2.png)

IV. Forcast: 1-step ahead prediction 1940-2016
----------------------------------------------

In 1-step ahead prediction of suicide rate for 1940-2016, for each predicted value in a certain year, it uses the data from previous years. For example, predict suicide rate in 1940 by using data from 1920 to 1939, similarly, predict suicide rate in 1941 by using data from 1920 to 1940, etc. In the following graph, the red tringles are suicide rate that are predicted from ARIMA(0,1,1). Two blue lines with circles are 95% confidence interval for the prediction values. And the black line with circles is the real suicide rates for each year. As the graph showed, overall, the model performs very good on predicting the suicide rate based on previous data set. However, not surprisingly, two significant prediction points fall off the 95% confidence interval.

``` r
pre.1step <- vector()
s.e       <- vector()
lowerCI   <- vector()
upperCI   <- vector()
estimate  <- vector()
alpha     <- 0.05

pred<-function(data,initial.box=19,prediction=0){
 
   for (i in 1: (nrow(suicide)-initial.box+prediction) ) {
     pre.data<-suicide[1:(initial.box+i),] ##16 year for first step
     ts.data <- ts(pre.data[,2] , start=c(1920), frequency=1)
     arma    <- arima(ts.data, order=c(0,1,1))
     
     pre.1step   <-predict(arma,n.ahead = 1)
     estimate[i] <- pre.1step$pred
     s.e[i]      <- pre.1step$se
     
     lowerCI[i] <- suicide.ts[20+i] - s.e[i]*qnorm(1-alpha/2)
     upperCI[i] <- suicide.ts[20+i] + s.e[i]*qnorm(1-alpha/2)
   }
  
  est.ts   <- ts(estimate,start = 1920+initial.box+1)
  cilow.ts <- ts(lowerCI,start = 1920+initial.box+1)
  ciup.ts  <- ts(upperCI,start = 1920+initial.box+1)
  avg.sd   <- mean(s.e)
  avg.diff <- mean( (estimate[1:76]-ts.data[21:96]) )

  table    <- data.frame(avg.sd,avg.diff)
  tablek   <- kable(table,booktabs=T) %>% kable_styling()
  plot(pre.data,type="o", xlim=c(1920,2016+prediction), ylim=c(6,18),
       cex=0.9, xaxt='n', ylab="suicide rate per 100,000",
       xlab="year", lty=1,main="1-Step Ahead Predictions for 1940-2016 by Using ARIMA(0,1,1)")
  axis(1, at=seq(1920,2016+prediction, by=5) )
  points(est.ts, col='red', pch=2, cex=0.8, lty=1)
  lines(cilow.ts, type='o', col='darkblue', cex=0.6, lty=1)
  lines(ciup.ts , type='o', col='darkblue', cex=0.6, lty=1)
  legend("topright", legend=c("Real Data","1-step ahead prediction","95% CI"),
         col=c("black", "red", "darkblue"), cex=0.7, lty=c(1,0,1), pch=c(1,2,1) )

  return(tablek)
  
  }

pred(suicide)
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-7-1.png)
<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
avg.sd
</th>
<th style="text-align:right;">
avg.diff
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
0.7427802
</td>
<td style="text-align:right;">
0.0212389
</td>
</tr>
</tbody>
</table>
V. Forcast: 10-steps ahead prediction 2016-2025
-----------------------------------------------

From the forecast, the next 10 years’ (2015-2025) U.S suicide rates (per100,000) will be 11.88087, 11.91215, 11.94344, 11.97472, 12.00600, 12.03728, 12.06856, 12.09985, 12.13113 and 12.16241. They all are prediced from the ARIMA(0,1,1) model based on the data from years 1920-2015.

``` r
library(sarima)
library(astsa)
sarima.for(sr, n.ahead=10, p=0, d=1, q=1, P = 0, D = 0, Q = 0, S =0)
```

![](Time-Series-Analysis-U.S.-Suicide-Rate_files/figure-markdown_github/unnamed-chunk-8-1.png)

    ## $pred
    ## Time Series:
    ## Start = 2016 
    ## End = 2025 
    ## Frequency = 1 
    ##  [1] 11.88087 11.91215 11.94344 11.97472 12.00600 12.03728 12.06856 12.09985
    ##  [9] 12.13113 12.16241
    ## 
    ## $se
    ## Time Series:
    ## Start = 2016 
    ## End = 2025 
    ## Frequency = 1 
    ##  [1] 0.5761048 1.0055429 1.3001294 1.5393363 1.7460745 1.9308019 2.0993370
    ##  [8] 2.2553127 2.4011779 2.5386759

VI. Summary
-----------

The understanding of important facts that affect suicide rates could helps the society be more aware of people's mental health during or before negative economic changes. In fact, the suicide rate (the U.S suicide rate in this case) may follow patterns which can be fitted into a model. A satisfactory model can also provide good forecasts of suicide rate in the furfure.

In the project, the ARIMA(0,1,1) model has been used to do two different type of forcasts. On one hand, the model looks good from some statistical examination or diagnostics. On the other hand, the model also ignores the natural complication of suicide. As time goes by, more factors that affect suidicte rate could be proved and known by scientist. Consiquently, there will be new factors to alter suicide rate. Therefore we need to always update our data, measure method, or even model. Indeed, the model in this project was idealistic, but it can still be a good reference for some institutions.
