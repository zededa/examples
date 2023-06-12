ifndef HZN_ORG_ID
$(error HZN_ORG_ID is not set. export HZN_ORG_ID=mycluster))
endif

ifndef HZN_EXCHANGE_USER_AUTH
$(error HZN_EXCHANGE_USER_AUTH is not set. export HZN_EXCHANGE_USER_AUTH=iamapikey:<your-iamapikey> )
endif
