"use strict";
cc._RF.push(module, '52d87+5JwxJVpcp4Szvn0th', 'leftToRight');
// 3.16小游戏/command_TypeScript/level1/leftToRight.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var mian_1 = require("./mian");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight = /** @class */ (function (_super) {
    __extends(leftToRight, _super);
    function leftToRight() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    leftToRight_1 = leftToRight;
    // LIFE-CYCLE CALLBACKS:
    leftToRight.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (mian_1.default.attack == true) {
            mian_1.default.minion1_hp -= 30;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        mian_1.default.attack = false;
    };
    leftToRight.prototype.onCollisionStay = function (other) {
        mian_1.default.attack = false;
        mian_1.default.minion_attack = false;
    };
    leftToRight.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (mian_1.default.minion1_hp <= 0) {
            other.node.active = false;
            mian_1.default.attack = false;
            mian_1.default.minion_attack = false;
            mian_1.default.main_hp = 163;
            mian_1.default.minion1_hp = 163;
            mian_1.default.minion2_hp = 163;
            mian_1.default.minion3_hp = 163;
            cc.director.loadScene("fight2");
        }
        else {
            other.node.children[0].setContentSize(mian_1.default.minion1_hp, 19);
        }
    };
    leftToRight.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight.prototype.update = function (dt) {
        if (mian_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
        leftToRight_1.current_x = this.node.position.x;
        leftToRight_1.current_y = this.node.position.y;
    };
    var leftToRight_1;
    __decorate([
        property(cc.Label)
    ], leftToRight.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight.prototype, "text", void 0);
    leftToRight = leftToRight_1 = __decorate([
        ccclass
    ], leftToRight);
    return leftToRight;
}(cc.Component));
exports.default = leftToRight;

cc._RF.pop();