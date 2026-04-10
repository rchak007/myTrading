#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * Setup (in core/execution/):
 *   npm install @7kprotocol/sdk-ts @mysten/sui@1 @flowx-finance/sdk @cetusprotocol/aggregator-sdk axios --legacy-peer-deps
 */

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const { coinIn, coinOut, amount, wallet, privateKey, slippage = "0.03", rpcUrl } = args;

    if (!coinIn || !coinOut || !amount || !wallet || !privateKey) {
        output({ status: "error", message: "Missing required args: --coinIn --coinOut --amount --wallet --privateKey" });
        process.exit(1);
    }

    try {
        const { SuiClient, getFullnodeUrl } = await import("@mysten/sui/client");
        const { Transaction } = await import("@mysten/sui/transactions");
        const sevenK = await import("@7kprotocol/sdk-ts");

        const suiClient = new SuiClient({ url: rpcUrl || getFullnodeUrl("mainnet") });
        const ma = new sevenK.MetaAg({ suiClient });

        // Step 0: Merge coins if fragmented
        output_log("Checking coin objects for merge...");
        await mergeCoinObjects(suiClient, wallet, coinIn, privateKey);

        // Step 1: Get quote
        output_log(`Getting quote: ${coinIn.split("::").pop()} -> ${coinOut.split("::").pop()} amount=${amount}`);
        const quotes = await ma.quote({ coinTypeIn: coinIn, coinTypeOut: coinOut, amountIn: amount });
        if (!quotes || quotes.length === 0) throw new Error("No quotes returned");

        const bestQuote = quotes[0];
        const expectedOut = bestQuote.amountOut || bestQuote.rawAmountOut || "?";
        output_log(`Quote received: provider=${bestQuote.provider} expectedOut=${expectedOut}`);

        // Step 2: Build transaction
        output_log("Building swap transaction...");
        const tx = new Transaction();

        // swap() returns the output coin reference — we must transfer it to the wallet
        const coinOutResult = await ma.swap({ signer: wallet, tx, quote: bestQuote });

        // Transfer the output coin to our wallet (SUI PTB requires all outputs to be consumed)
        if (coinOutResult) {
            tx.transferObjects([coinOutResult], wallet);
        }

        // Step 3: Sign
        output_log("Signing...");
        const keypair = await parseKeypair(privateKey);
        tx.setSender(wallet);
        const txBytes = await tx.build({ client: suiClient });
        const { signature } = await keypair.signTransaction(txBytes);

        // Step 4: Execute
        output_log("Executing...");
        const result = await suiClient.executeTransactionBlock({
            transactionBlock: txBytes,
            signature: [signature],
            options: { showEffects: true, showEvents: true },
        });

        const digest = result.digest;
        const status = result.effects?.status?.status;

        if (status === "success") {
            output({ status: "success", digest, amountOut: String(expectedOut) });
        } else {
            const error = result.effects?.status?.error || "unknown";
            output({ status: "error", message: `Transaction failed on-chain: ${error}`, digest });
            process.exit(1);
        }
    } catch (err) {
        output({
            status: "error",
            message: err.message || String(err),
            stack: err.stack ? err.stack.split("\n").slice(0, 5).join(" | ") : undefined,
        });
        process.exit(1);
    }
}

async function mergeCoinObjects(suiClient, wallet, coinType, privateKey) {
    const { Transaction } = await import("@mysten/sui/transactions");

    const allCoins = [];
    let cursor = null;
    while (true) {
        const page = await suiClient.getCoins({ owner: wallet, coinType, cursor });
        allCoins.push(...page.data);
        if (!page.hasNextPage) break;
        cursor = page.nextCursor;
    }

    const nonZero = allCoins.filter(c => BigInt(c.balance) > 0n);
    output_log(`Found ${allCoins.length} coin objects (${nonZero.length} non-zero) for ${coinType.split("::").pop()}`);

    if (nonZero.length <= 1) {
        output_log("No merge needed");
        return;
    }

    output_log(`Merging ${nonZero.length} coin objects into one...`);
    const tx = new Transaction();
    tx.mergeCoins(tx.object(nonZero[0].coinObjectId), nonZero.slice(1).map(c => tx.object(c.coinObjectId)));

    const keypair = await parseKeypair(privateKey);
    tx.setSender(wallet);
    const txBytes = await tx.build({ client: suiClient });
    const { signature } = await keypair.signTransaction(txBytes);

    const result = await suiClient.executeTransactionBlock({
        transactionBlock: txBytes, signature: [signature], options: { showEffects: true },
    });

    if (result.effects?.status?.status !== "success") {
        throw new Error(`Coin merge failed: ${result.effects?.status?.error || "unknown"}`);
    }
    output_log(`Merge successful: digest=${result.digest}`);
    await new Promise(r => setTimeout(r, 2000));
}

async function parseKeypair(privateKey) {
    const { Ed25519Keypair } = await import("@mysten/sui/keypairs/ed25519");
    if (privateKey.startsWith("suiprivkey")) {
        const { decodeSuiPrivateKey } = await import("@mysten/sui/cryptography");
        const parsed = decodeSuiPrivateKey(privateKey);
        return Ed25519Keypair.fromSecretKey(parsed.secretKey);
    }
    return Ed25519Keypair.fromSecretKey(Buffer.from(privateKey.replace("0x", ""), "hex"));
}

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith("--") && i + 1 < argv.length) { args[argv[i].slice(2)] = argv[i + 1]; i++; }
    }
    return args;
}

function output(obj) { console.log(JSON.stringify(obj)); }
function output_log(msg) { process.stderr.write(`[sui_swap_helper] ${msg}\n`); }

main();