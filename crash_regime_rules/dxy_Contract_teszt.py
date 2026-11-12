from ib_insync import IB, ContFuture

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=51)

dx_contract = ContFuture("DX", "NYBOT", "USD")

contracts = ib.qualifyContracts(dx_contract)

for contract in contracts:
    print(
        contract.localSymbol,
        contract.symbol,
        contract.exchange,
        contract.conId,
    )

ib.disconnect()