# Code: Harrison Satcher
# PI: Adam Morris
# Cushman Lab
# Model Free Punishments
# Created: 10/31/17
# Last modified: 10/31/17

if (!require("lme4")) {install.packages("lme4"); require("lme4")}
if (!require("lmerTest")) {install.packages("lmerTest"); require("lmerTest")}

dat_long = read.csv("long.csv")
dat_short = read.csv("short.csv")
colnames(dat_long) <- c("ShortLong", "AgentNumber", "Reward", "StayGo")
colnames(dat_short) <- c("ShortLong", "AgentNumber", "Reward", "StayGo")
dat <- rbind(dat_long,dat_short)
# Data not centered, could be causing issues
outcomeModel <- glmer(StayGo ~ ShortLong*Reward*(1+Reward|AgentNumber), data = dat, family=binomial())
summary(outcomeModel)
